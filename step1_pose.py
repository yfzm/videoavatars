#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import h5py
import argparse
import numpy as np
import chumpy as ch
import cPickle as pkl

from opendr.camera import ProjectPoints
from opendr.lighting import LambertianPointLight
from opendr.renderer import ColoredRenderer
from opendr.filters import gaussian_pyramid

from util import im
from util.logger import log
from lib.frame import FrameData
from models.smpl import Smpl, copy_smpl, joints_coco
from models.bodyparts import faces_no_hands

from vendor.smplify.sphere_collisions import SphereCollisions
from vendor.smplify.robustifiers import GMOf
# from multiprocessing import Pool
from multiprocessing import Process

THREAD_NUM = 2


def collision_obj(smpl, regs):
    sp = SphereCollisions(pose=smpl.pose, betas=smpl.betas, model=smpl, regs=regs)
    sp.no_hands = True

    return sp


def pose_prior_obj(smpl, prior_data):
    return (smpl.pose[3:] - prior_data['mean']).reshape(1, -1).dot(prior_data['prec'])


def height_predictor(b2m, betas):
    return ch.hstack((betas.reshape(1, -1), [[1]])).dot(b2m)


# def handle_thread(inst, thread_num):
#     inst.handle_thread_inner(thread_num)


class Step1:
    def __init__(self, keypoint_file, masks_file, camera_file, out, model_file, prior_file, resize,
                 body_height, nohands, display):
        # load data
        with open(model_file, 'rb') as fp:
            self.model_data = pkl.load(fp)

        with open(camera_file, 'rb') as fp:
            self.camera_data = pkl.load(fp)

        with open(prior_file, 'rb') as fp:
            self.prior_data = pkl.load(fp)

        if 'basicModel_f' in model_file:
            self.regs = np.load('vendor/smplify/models/regressors_locked_normalized_female.npz')
            self.b2m = np.load('assets/b2m_f.npy')
        else:
            self.regs = np.load('vendor/smplify/models/regressors_locked_normalized_male.npz')
            self.b2m = np.load('assets/b2m_m.npy')

        self.keypoints = h5py.File(keypoint_file, 'r')['keypoints']
        self.masks = h5py.File(masks_file, 'r')['masks']
        self.num_frames = self.masks.shape[0]
        self.resize = resize
        self.body_height = body_height
        self.nohands = nohands
        self.out = out

        # init
        self.base_smpl = Smpl(self.model_data)
        self.base_smpl.trans[:] = np.array([0, 0, 3])
        self.base_smpl.pose[0] = np.pi
        self.base_smpl.pose[3:] = self.prior_data['mean']

        self.camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=self.camera_data['camera_c'] * resize,
                                    f=self.camera_data['camera_f'] * resize, k=self.camera_data['camera_k'],
                                    v=self.base_smpl)
        self.frustum = {'near': 0.1, 'far': 1000.,
                        'width': int(self.camera_data['width'] * resize),
                        'height': int(self.camera_data['height'] * resize)}

        if display:
            self.debug_cam = ProjectPoints(v=self.base_smpl, t=self.camera.t, rt=self.camera.rt, c=self.camera.c,
                                           f=self.camera.f, k=self.camera.k)
            self.debug_light = LambertianPointLight(f=self.base_smpl.f, v=self.base_smpl, num_verts=len(self.base_smpl),
                                                    light_pos=np.zeros(3),
                                                    vc=np.ones(3), light_color=np.ones(3))
            self.debug_rn = ColoredRenderer(camera=self.debug_cam, v=self.base_smpl, f=self.base_smpl.f,
                                            vc=self.debug_light,
                                            frustum=self.frustum)
        else:
            self.debug_rn = None

        self.base_frame = self.create_frame(0, self.base_smpl, copy=False)
        self.fp = h5py.File(self.out, 'w')
        self.poses_dset = self.fp.create_dataset("pose", (self.num_frames, 72), 'f', chunks=True, compression="lzf")
        self.trans_dset = self.fp.create_dataset("trans", (self.num_frames, 3), 'f', chunks=True, compression="lzf")
        self.betas_dset = self.fp.create_dataset("betas", (10,), 'f', chunks=True, compression="lzf")

        num_init = 5
        indices_init = np.ceil(np.arange(num_init) * self.num_frames * 1. / num_init).astype(np.int)

        init_frames = [self.base_frame]
        for i in indices_init[1:]:
            init_frames.append(self.create_frame(i, self.base_smpl))

        self.init(init_frames, self.body_height, self.b2m, self.debug_rn)

    def __del__(self):
        self.fp.close()

    def init(self, frames, body_height, b2m, viz_rn):
        betas = frames[0].smpl.betas

        E_height = None
        if body_height is not None:
            E_height = height_predictor(b2m, betas) - body_height * 1000.

        # first get a rough pose for all frames individually
        for i, f in enumerate(frames):
            if np.sum(f.keypoints[[0, 2, 5, 8, 11], 2]) > 3.:
                if f.keypoints[2, 0] > f.keypoints[5, 0]:
                    f.smpl.pose[0] = 0
                    f.smpl.pose[2] = np.pi

                E_init = {
                    'init_pose_{}'.format(i): f.pose_obj[[0, 2, 5, 8, 11]]
                }

                x0 = [f.smpl.trans, f.smpl.pose[:3]]

                if E_height is not None and i == 0:
                    E_init['height'] = E_height
                    E_init['betas'] = betas
                    x0.append(betas)

                ch.minimize(
                    E_init,
                    x0,
                    method='dogleg',
                    options={
                        'e_3': .01,
                    },
                    callback=None
                    # callback = self.get_cb(viz_rn, f)
                )

        weights = zip(
            [5., 4.5, 4.],
            [5., 4., 3.]
        )

        E_betas = betas - betas.r

        for w_prior, w_betas in weights:
            x0 = [betas]

            E = {
                'betas': E_betas * w_betas,
            }

            if E_height is not None:
                E['height'] = E_height

            for i, f in enumerate(frames):
                if np.sum(f.keypoints[[0, 2, 5, 8, 11], 2]) > 3.:
                    x0.extend([f.smpl.pose[range(21) + range(27, 30) + range(36, 60)], f.smpl.trans])
                    E['pose_{}'.format(i)] = f.pose_obj
                    E['prior_{}'.format(i)] = f.pose_prior_obj * w_prior

            ch.minimize(
                E,
                x0,
                method='dogleg',
                options={
                    'e_3': .01,
                },
                callback=None  # self.get_cb(viz_rn, frames[0])
            )

    def reinit_frame(self, frame, null_pose, nohands, viz_rn):
        if (np.sum(frame.pose_obj.r ** 2) > 625 or np.sum(frame.pose_prior_obj.r ** 2) > 75) \
                and np.sum(frame.keypoints[[0, 2, 5, 8, 11], 2]) > 3.:

            log.info('Tracking error too large. Re-init frame...')

            x0 = [frame.smpl.pose[:3], frame.smpl.trans]

            frame.smpl.pose[3:] = null_pose
            if frame.keypoints[2, 0] > frame.keypoints[5, 0]:
                frame.smpl.pose[0] = 0
                frame.smpl.pose[2] = np.pi

            E = {
                'init_pose': frame.pose_obj[[0, 2, 5, 8, 11]],
            }

            ch.minimize(
                E,
                x0,
                method='dogleg',
                options={
                    'e_3': .1,
                },
                callback=None  # self.get_cb(viz_rn, frame)
            )

            E = {
                'pose': GMOf(frame.pose_obj, 100),
                'prior': frame.pose_prior_obj * 8.,
            }

            x0 = [frame.smpl.trans]

            if nohands:
                x0.append(frame.smpl.pose[range(21) + range(27, 30) + range(36, 60)])
            else:
                x0.append(frame.smpl.pose[range(21) + range(27, 30) + range(36, 72)])

            ch.minimize(
                E,
                x0,
                method='dogleg',
                options={
                    'e_3': .01,
                },
                callback=None  # self.get_cb(viz_rn, frame)
            )

    def fit_pose(self, frame, last_smpl, frustum, nohands, viz_rn):
        if nohands:
            faces = faces_no_hands(frame.smpl.f)
        else:
            faces = frame.smpl.f

        dst_type = cv2.cv.CV_DIST_L2 if cv2.__version__[0] == '2' else cv2.DIST_L2

        dist_i = cv2.distanceTransform(np.uint8(frame.mask * 255), dst_type, 5) - 1
        dist_i[dist_i < 0] = 0
        dist_i[dist_i > 50] = 50
        dist_o = cv2.distanceTransform(255 - np.uint8(frame.mask * 255), dst_type, 5)
        dist_o[dist_o > 50] = 50

        rn_m = ColoredRenderer(camera=frame.camera, v=frame.smpl, f=faces, vc=np.ones_like(frame.smpl), frustum=frustum,
                               bgcolor=0, num_channels=1)

        E = {
            'mask': gaussian_pyramid(rn_m * dist_o * 100. + (1 - rn_m) * dist_i, n_levels=4,
                                     normalization='size') * 80.,
            '2dpose': GMOf(frame.pose_obj, 100),
            'prior': frame.pose_prior_obj * 4.,
            'sp': frame.collision_obj * 1e3,
        }

        if last_smpl is not None:
            E['last_pose'] = GMOf(frame.smpl.pose - last_smpl.pose, 0.05) * 50.
            E['last_trans'] = GMOf(frame.smpl.trans - last_smpl.trans, 0.05) * 50.

        if nohands:
            x0 = [frame.smpl.pose[range(21) + range(27, 30) + range(36, 60)], frame.smpl.trans]
        else:
            x0 = [frame.smpl.pose[range(21) + range(27, 30) + range(36, 72)], frame.smpl.trans]

        ch.minimize(
            E,
            x0,
            method='dogleg',
            options={
                'e_3': .01,
            },
            callback=None
            # callback = self.get_cb(viz_rn, frame)
        )

    def get_cb(self, viz_rn, f):
        if viz_rn is not None:
            viz_rn.set(v=f.smpl, background_image=np.dstack((f.mask, f.mask, f.mask)))
            viz_rn.vc.set(v=f.smpl)

            def cb(_):
                debug = np.array(viz_rn.r)

                for j in f.J_proj.r:
                    cv2.circle(debug, tuple(j.astype(np.int)), 3, (0, 0, 0.8), -1)
                for j in f.keypoints[:, :2]:
                    cv2.circle(debug, tuple(j.astype(np.int)), 3, (0, 0.8, 0), -1)

                im.show(debug, id='pose', waittime=1)
        else:
            cb = None

        return cb

    # generic frame loading function
    def create_frame(self, i, smpl, copy=True):
        f = FrameData()

        f.smpl = copy_smpl(smpl, self.model_data) if copy else smpl
        f.camera = ProjectPoints(v=f.smpl, t=self.camera.t, rt=self.camera.rt, c=self.camera.c, f=self.camera.f,
                                 k=self.camera.k)

        f.keypoints = np.array(self.keypoints[i]).reshape(-1, 3) * np.array([self.resize, self.resize, 1])
        f.J = joints_coco(f.smpl)
        f.J_proj = ProjectPoints(v=f.J, t=self.camera.t, rt=self.camera.rt, c=self.camera.c, f=self.camera.f,
                                 k=self.camera.k)
        f.mask = cv2.resize(np.array(self.masks[i], dtype=np.float32), (0, 0),
                            fx=self.resize, fy=self.resize, interpolation=cv2.INTER_NEAREST)

        f.collision_obj = collision_obj(f.smpl, self.regs)
        f.pose_prior_obj = pose_prior_obj(f.smpl, self.prior_data)
        f.pose_obj = (f.J_proj - f.keypoints[:, :2]) * f.keypoints[:, 2].reshape(-1, 1)

        return f

    def handle_thread_inner(self, thread_num):
        last_smpl = None
        begin = thread_num * self.num_frames / THREAD_NUM
        end = (thread_num + 1) * self.num_frames / THREAD_NUM
        for i in xrange(begin, end):
            if i == begin:
                current_frame = self.base_frame
            else:
                current_frame = self.create_frame(i, last_smpl)

            log.info('Fit frame {}'.format(i))
            # re-init if necessary
            self.reinit_frame(current_frame, self.prior_data['mean'], self.nohands, self.debug_rn)
            # final fit
            self.fit_pose(current_frame, last_smpl, self.frustum, self.nohands, self.debug_rn)

            self.poses_dset[i] = current_frame.smpl.pose.r
            self.trans_dset[i] = current_frame.smpl.trans.r

            if i == begin:
                self.betas_dset[:] = current_frame.smpl.betas.r

            last_smpl = current_frame.smpl

    def run(self):

        # get betas from 5 frames
        log.info('Initial fit')

        # num_init = 5
        # indices_init = np.ceil(np.arange(num_init) * self.num_frames * 1. / num_init).astype(np.int)
        #
        # init_frames = [self.base_frame]
        # for i in indices_init[1:]:
        #     init_frames.append(self.create_frame(i, self.base_smpl))
        #
        # init(init_frames, self.body_height, self.b2m, self.debug_rn)

        # fp = h5py.File(self.out, 'w')
        # self.poses_dset = fp.create_dataset("pose", (self.num_frames, 72), 'f', chunks=True, compression="lzf")
        # self.trans_dset = fp.create_dataset("trans", (self.num_frames, 3), 'f', chunks=True, compression="lzf")
        # self.betas_dset = fp.create_dataset("betas", (10,), 'f', chunks=True, compression="lzf")

        # pool = multiprocessing.Pool(THREAD_NUM)
        # pool.map(handle_thread, [(self, i) for i in range(THREAD_NUM)])
        # pool.join()
        ps = []
        for i in range(THREAD_NUM):
            p = Process(target=self.handle_thread_inner, args=(i,))
            p.start()
            ps.append(p)
            # thread.start_new_thread(self.handle_thread_inner, (i,))
            # pool.apply_async(self.handle_thread_inner, (i,))
        # pool.close()
        # pool.join()
        for p in ps:
            p.join()
        log.info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'keypoint_file',
        type=str,
        help="File that contains 2D keypoint detections")
    parser.add_argument(
        'masks_file',
        type=str,
        help="File that contains segmentations")
    parser.add_argument(
        'camera',
        type=str,
        help="pkl file that contains camera settings")
    parser.add_argument(
        'out',
        type=str,
        help="Out file path")
    parser.add_argument(
        '--model', '-m',
        default='vendor/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl',
        help='Path to SMPL model')
    parser.add_argument(
        '--prior', '-p',
        default='assets/prior_a_pose.pkl',
        help='Path to pose prior')
    parser.add_argument(
        '--resize', '-r', default=0.5, type=float,
        help="Resize factor")
    parser.add_argument(
        '--body_height', '-bh', default=None, type=float,
        help="Height of the subject in meters (optional)")
    parser.add_argument(
        '--nohands', '-nh',
        action='store_true',
        help="Exclude hands from optimization")
    parser.add_argument(
        '--display', '-d',
        action='store_true',
        help="Enable visualization")

    args = parser.parse_args()

    step1 = Step1(args.keypoint_file, args.masks_file, args.camera, args.out, args.model, args.prior, args.resize,
                  args.body_height, args.nohands, args.display)
    print "display? {}".format(args.display)
    step1.run()

    # for i in range(THREAD_NUM):
    #     thread.start_new_thread(handle_thread, (step1, i))
    # pool = Pool(THREAD_NUM)
    # pool.map(handle_thread, ((step1, i) for i in range(THREAD_NUM)))
    # pool.close()
    # pool.join()
    log.info('Done.')
    while 1:
        pass
