import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageEnhance

class trimesh_renderer():
    def __init__(self, img_w, img_h):
        self.img_w = img_w
        self.img_h = img_h
        self.default_focal = 500
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                       viewport_height=img_h,
                                       point_size=1.0)
    
    def __call__(self, fuze_trimesh, rot=None, t=None, image=None, 
                 fx=None, fy=None, cx=None, cy=None, mask_over=False):
        
        # Camera parameter
        if fx is None or fy is None:
            fx = self.default_focal
            fy = self.default_focal
            
        if cx is None or cy is None:
            cx = self.img_w / 2
            cy = self.img_h / 2
        
        # 6DoF object pose in camera coordinate
        # You can skip this and apply directly on the input mesh
        if rot is None:
            rot = np.eye(3)
        if t is None:
            t = np.zeros(3)
        transform = np.zeros([4,4])
        transform[:3, :3] = rot
        transform[:3, -1] = t
        fuze_trimesh.apply_transform(transform)


        # OpenGL convension
        transform = trimesh.transformations.rotation_matrix(
                    np.radians(180), [1, 0, 0])
        fuze_trimesh.apply_transform(transform)

    
        mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh)

        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=3000)
        camera_pose = np.eye(4)
        scene.add(camera, pose=camera_pose)
        
        # Render
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.uint8)
        
        if image is None:
            return color
        

        valid_mask = (rend_depth>0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)

        if mask_over:
            mask = np.zeros([self.img_h, self.img_w, 3])
            mask[:,:,1] = 250
            alpha = 0.3
            mask = alpha * mask + (1-alpha) * image
            overlay = mask * valid_mask + image * (1-valid_mask)
            overlay = overlay.astype(np.uint8)
            return output_img, overlay

        else:
            return output_img

    def render_scene(self, meshes, Rs, ts, image=None, fx=None, fy=None, cx=None, cy=None):
        
        # Camera parameter
        if fx is None or fy is None:
            fx = self.default_focal
            fy = self.default_focal
            
        if cx is None or cy is None:
            cx = self.img_w / 2
            cy = self.img_h / 2

        color = (0.2, 0.4, 0.2, 1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            alphaMode='OPAQUE',
            baseColorFactor=color)

        # 6DoF object pose in camera coordinate
        # You can skip this and apply directly on the input mesh
        for i, mesh in enumerate(meshes):
            transform = np.zeros([4,4])
            transform[:3, :3] = Rs[i]
            transform[:3, -1] = ts[i]
            mesh.apply_transform(transform)


        # OpenGL convension
        transform = trimesh.transformations.rotation_matrix(
                    np.radians(180), [1, 0, 0])
        for mesh in meshes:
            mesh.apply_transform(transform)


        scene = pyrender.Scene(ambient_light=(0.8, 0.8, 0.8, 1.0))
        for fuze_trimesh in meshes:
            mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
            scene.add(mesh)

        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=3000)
        camera_pose = np.eye(4)
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)
        
        # Render
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.uint8)
        
        enhancer = ImageEnhance.Contrast(Image.fromarray(color))
        factor = 1.2 #increase contrast
        color = enhancer.enhance(factor)
        color = np.array(color, dtype=np.uint8)
        
        if image is None:
            return color
        

        valid_mask = (rend_depth>0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)

        return output_img

