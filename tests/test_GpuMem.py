#
# Copyright 2024 Yves33
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Starting from Python 3.8 DLL search policy has changed.
# We need to add path to CUDA DLLs explicitly.
import sys
import os
from os.path import join, dirname

__DBG_IMG__ = False

if os.name == "nt":
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file=sys.stderr)
        print("Can't set CUDA DLLs search path.", file=sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(";")
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file=sys.stderr)
        exit(1)

import PyNvCodec as nvc
import numpy as np
import unittest
import json
import test_common as tc
try:
    import pycuda
    import pycuda.autoinit
    PYCUDA_AVAIL=True
except:
    print("Pycuda is required to run this test")
    PYCUDA_AVAIL=False

try:
    sys.path.insert(0,"../extern/dlpack/apps/numpy_dlpack/dlpack")
    from dlpack import _c_str_dltensor, DLManagedTensor
    DLPACK_AVAIL=True
except:
    DLPACK_AVAIL=False

tolerance=0.0

## rather than copying data to host for testing GpuMem(),
## one can also compare with the value stored inside pycapsule returned by __dlpack__
import ctypes
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

def CapsulePtr(capsule):
    dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(capsule, _c_str_dltensor)
    dl_managed_tensor_ptr = ctypes.cast(dl_managed_tensor, ctypes.POINTER(DLManagedTensor))
    return dl_managed_tensor_ptr.contents.dl_tensor.data
  
def dbg_img(buffer,w,h,format,merge=False):
    """
    utility class to display buffers. requires numpy, pillow.
    adapt code to use non blocking image.show()
    
    Parameters
    ----------
    buffer:  Numpy array
    w, h :   Width and height of image, regardeless of buffer format
    format : Format of buffer. may be RGB, RGB_Planar, NV12, YUV420p
    merge :  Whether we should reconstitute rgb image or display the whole buffer (as grayscale image)
    """
    if not __DBG_IMG__:
        return
    try:
        from PIL import Image,ImageShow
        
        #linux/unix only. forces image.show() to block
        class EomViewer(ImageShow.UnixViewer):
            def show_file(self, filename, **options):
                os.system('eom %s' % filename)
                return 1
        ImageShow.register(EomViewer, order=-1)
        
    except:
        return
    match format:
        case 'RGB':
            Image.fromarray(buffer.reshape(h,w,3)).show()
        case 'RGB_Planar':
            if not merge:
                Image.fromarray(buffer.reshape(h*3,w)).show()
            else:
                Image.fromarray(buffer.reshape(3,h,w).transpose(2,1,0)).transpose(Image.TRANSPOSE).show()
        case 'NV12':
            if not merge:
                Image.fromarray(buffer.reshape(h*3//2,w)).show()
            else:
                y=Image.fromarray(buffer[:w*h].reshape(h,w))
                u=Image.fromarray(buffer[w*h:][::2].reshape(h//2,w//2).repeat(2, axis=0).repeat(2, axis=1))
                v=Image.fromarray(buffer[w*h+1:][::2].reshape(h//2,w//2).repeat(2, axis=0).repeat(2, axis=1))
                Image.merge('YCbCr', (y, u,v)).show()
        case 'YUV420':
            if not merge:
                Image.fromarray(buffer.reshape(h*3//2,w)).show()
            else:
                y=Image.fromarray(buffer[:w*h].reshape(h,w))
                u=Image.fromarray(buffer[w*h:w*h*5//4].reshape(h//2,w//2).repeat(2, axis=0).repeat(2, axis=1))
                v=Image.fromarray(buffer[w*h*5//4:].reshape(h//2,w//2).repeat(2, axis=0).repeat(2, axis=1))
                Image.merge('YCbCr', (y,u,v)).show()

class TestSurfaceConverter(unittest.TestCase):
    """
    First test:
    for each test in [basic_rgb,basic_rgb_planar, basic_nv12, basic_hevc10_p10]
    + read etalon
    + create surface
    + copy data to device surface (numpy->device)
    + create second surface
    + copy first surface to second surface (device->device)
    + dowload to host (device->numpy)
    + compare resulting surface with original surface
    + for both surface, checks that the value of GpuMem() is the same as the one in pycapsule returned by dlpack

    Second test
    for each test in [basic, basic_mpeg4, res_change, hevc10]:
    + decodes surfaces from uri
    + creates both yuv420 and nv12 surfaces from raw decoded surface using hardware converters
    + download nv12 surface plane[0] to host
    + download yuv surface planes to host (separately)
    + compares individual y, u, v planes with correct portion of nv12 plane      
    """

    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        self.pixelsize={"basic_rgb":(3,1),
                        "basic_rgb_planar":(3,1),
                        "basic_nv12":(3,2),
                        "hevc10_p10":(3,2)
                        }
        self.px_format={"basic_rgb":nvc.PixelFormat.RGB,
                        "basic_rgb_planar":nvc.PixelFormat.RGB_PLANAR,
                        "basic_nv12":nvc.PixelFormat.NV12,
                        "hevc10_p10":nvc.PixelFormat.P10
                        }

    def _test_copy(self, selection):
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            gt_info = tc.GroundTruth(**gt_values[selection])
            px_num,px_den=self.pixelsize[selection]
            px_format= self.px_format[selection]
            f_in = open(gt_info.uri, "rb")
            for i in range(0, gt_info.num_frames):
                frame_size = gt_info.width * gt_info.height * px_num // px_den
                # Read from ethalon RGB file
                gt_frame = np.fromfile(file=f_in, dtype=np.uint8, count=frame_size)
                # Host to GPU
                surf_rgb = nvc.Surface.Make(px_format, gt_info.width, 
                                            gt_info.height, gpu_id=0)
                host_to_device=pycuda.driver.Memcpy2D()
                host_to_device.set_src_host(gt_frame)
                host_to_device.set_dst_device(surf_rgb.PlanePtr(0).GpuMem())
                host_to_device.width_in_bytes = len(gt_frame)//surf_rgb.PlanePtr(0).Height()
                host_to_device.src_pitch = len(gt_frame)//surf_rgb.PlanePtr(0).Height()
                host_to_device.src_height = surf_rgb.PlanePtr(0).Height()
                host_to_device.dst_pitch = surf_rgb.PlanePtr(0).Pitch()
                host_to_device.height = surf_rgb.PlanePtr(0).Height()
                host_to_device(aligned=False)
                
                ## GPU to GPU. required for openGL interop
                surf_rgb_2 = nvc.Surface.Make(px_format, surf_rgb.Width(), 
                                            surf_rgb.Height(), gpu_id=0)
                device_to_device=pycuda.driver.Memcpy2D()
                device_to_device.set_src_device(surf_rgb.PlanePtr(0).GpuMem())
                device_to_device.set_dst_device(surf_rgb_2.PlanePtr(0).GpuMem())
                device_to_device.width_in_bytes = surf_rgb.PlanePtr(0).Width()
                device_to_device.src_pitch = surf_rgb.PlanePtr(0).Pitch()
                device_to_device.dst_pitch = surf_rgb_2.PlanePtr(0).Pitch()
                device_to_device.src_height = surf_rgb.PlanePtr(0).Height()
                device_to_device.height = surf_rgb_2.PlanePtr(0).Height()
                device_to_device(aligned=False)

                ## GPU to host
                frame_size = surf_rgb_2.Width()*surf_rgb_2.Height() * px_num // px_den
                host_frame = np.zeros(shape=frame_size, dtype=np.uint8)
                device_to_host=pycuda.driver.Memcpy2D()
                device_to_host.set_src_device(surf_rgb_2.PlanePtr(0).GpuMem())
                device_to_host.set_dst_host(host_frame)
                device_to_host.width_in_bytes = surf_rgb_2.PlanePtr(0).Width()
                device_to_host.src_pitch = surf_rgb_2.PlanePtr(0).Pitch()
                device_to_host.dst_pitch = surf_rgb_2.PlanePtr(0).Width()
                device_to_host.src_height = surf_rgb_2.PlanePtr(0).Height()
                device_to_host.height = surf_rgb_2.PlanePtr(0).Height()
                device_to_host(aligned=False)
                
                self.assertTrue (surf_rgb.PlanePtr(0).GpuMem()!=surf_rgb_2.PlanePtr(0).GpuMem())
                if DLPACK_AVAIL:
                    self.assertEqual(surf_rgb.PlanePtr(0).GpuMem(),CapsulePtr(surf_rgb.PlanePtr(0).__dlpack__()))
                    self.assertEqual(surf_rgb_2.PlanePtr(0).GpuMem(),CapsulePtr(surf_rgb_2.PlanePtr(0).__dlpack__()))
                self.assertTrue((host_frame==gt_frame).all())
                match selection:
                    case "basic_rgb":
                        dbg_img(host_frame,surf_rgb_2.Width(),surf_rgb_2.Height(),"RGB",merge=True)
                    case "basic_rgb_planar":
                        dbg_img(host_frame,surf_rgb_2.Width(),surf_rgb_2.Height(),"RGB_Planar",merge=True)
                    case "basic_nv12":
                        dbg_img(host_frame,surf_rgb_2.Width(),surf_rgb_2.Height(),"NV12",merge=True)
            f_in.close()

    def _test_yuv(self, selection):
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            gt_info = tc.GroundTruth(**gt_values[selection])
        
        nvDec = nvc.PyDecoder(gt_info.uri, opts={}, gpu_id=0)
        nvCvt_p10_nv12=nvc.PySurfaceConverter( nvc.PixelFormat.P10, nvc.PixelFormat.NV12,gpu_id=0)
        nvCvt_nv12_yuv420=nvc.PySurfaceConverter( nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420,gpu_id=0)
        nvCvt_yuv420_nv12=nvc.PySurfaceConverter( nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12,gpu_id=0)
        
        ccCtx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_709, nvc.ColorRange.MPEG)
        w,h=nvDec.Width(), nvDec.Height()
        for i in range(0, gt_info.num_frames):
            surf_raw=nvc.Surface.Make(nvDec.Format(),nvDec.Width(), nvDec.Height(),gpu_id=0)
            success,info=nvDec.DecodeSingleSurface(surf_raw)
            if ( w!=nvDec.Width() or h!=nvDec.Height()):
                print("Resolution changed...Adjusting resolutions...")
                w=nvDec.Width()
                h=nvDec.Height()
                continue
            
            if surf_raw.Format()==nvc.PixelFormat.YUV420:
                surf_yuv=surf_raw
                surf_nv12=nvc.Surface.Make(nvc.PixelFormat.NV12,nvDec.Width(), nvDec.Height(),gpu_id=0)
                success, info=nvCvt_yuv420_nv12.Run(surf_yuv,surf_nv12,ccCtx)
            elif surf_raw.Format()==nvc.PixelFormat.NV12:
                surf_nv12=surf_raw
                surf_yuv=nvc.Surface.Make(nvc.PixelFormat.YUV420,nvDec.Width(), nvDec.Height(),gpu_id=0)
                success, info=nvCvt_nv12_yuv420.Run(surf_nv12,surf_yuv,ccCtx)
            elif surf_raw.Format()==nvc.PixelFormat.P10:
                surf_nv12=nvc.Surface.Make(nvc.PixelFormat.NV12,nvDec.Width(), nvDec.Height(),gpu_id=0)
                surf_yuv=nvc.Surface.Make(nvc.PixelFormat.YUV420,nvDec.Width(), nvDec.Height(),gpu_id=0)
                success, info=nvCvt_p10_nv12.Run(surf_raw,surf_nv12,ccCtx)
                success, info=nvCvt_nv12_yuv420.Run(surf_nv12,surf_yuv,ccCtx)
            else:
                print("Unhandled pixel format")
            
            pycuda.driver.Context.synchronize() ## wait for convertions to finish before downloading to host
            
            ## download nv12 plane
            frame_size = surf_nv12.Width()*surf_nv12.Height() * 3//2
            nv12_frame = np.zeros(shape=frame_size, dtype=np.uint8)
            device_to_host=pycuda.driver.Memcpy2D()
            device_to_host.set_src_device(surf_nv12.PlanePtr(0).GpuMem())
            device_to_host.set_dst_host(nv12_frame)
            device_to_host.width_in_bytes = surf_nv12.PlanePtr(0).Width()
            device_to_host.src_pitch = surf_nv12.PlanePtr(0).Pitch()
            device_to_host.dst_pitch = surf_nv12.PlanePtr(0).Width()
            device_to_host.src_height = surf_nv12.PlanePtr(0).Height()
            device_to_host.height = surf_nv12.PlanePtr(0).Height()
            device_to_host(aligned=False)

            ## download individual Y, u, v planes
            frame_sizes=[surf_nv12.Width()*surf_nv12.Height(),
                        surf_nv12.Width()*surf_nv12.Height()//4,
                        surf_nv12.Width()*surf_nv12.Height()//4]
            yuv_planes=[np.zeros(shape=fsz,dtype=np.uint8) for fsz in frame_sizes]
            for pl in range(surf_yuv.NumPlanes()):
                device_to_host=pycuda.driver.Memcpy2D()
                device_to_host.set_src_device(surf_yuv.PlanePtr(pl).GpuMem())
                device_to_host.set_dst_host(yuv_planes[pl])
                device_to_host.width_in_bytes = surf_yuv.PlanePtr(pl).Width()
                device_to_host.src_pitch = surf_yuv.PlanePtr(pl).Pitch()
                device_to_host.dst_pitch = surf_yuv.PlanePtr(pl).Width()
                device_to_host.src_height = surf_yuv.PlanePtr(pl).Height()
                device_to_host.height = surf_yuv.PlanePtr(pl).Height()
                device_to_host(aligned=False)
            sz=surf_nv12.Width()*surf_nv12.Height()
            #dbg_img(np.concatenate(yuv_planes),surf_nv12.Width(),surf_nv12.Height(),'YUV420',merge=True)
            #dbg_img(nv12_frame,surf_nv12.Width(),surf_nv12.Height(),'NV12',merge=True)
            assert(np.allclose(nv12_frame[:sz], yuv_planes[0], atol=tolerance))
            assert(np.allclose(nv12_frame[sz:][::2],yuv_planes[1],atol=tolerance))
            assert(np.allclose(nv12_frame[sz+1:][::2],yuv_planes[2],atol=tolerance))

    def test_copy_rgb(self):
        self._test_copy("basic_rgb")

    def test_copy_rgb_planar(self):
        self._test_copy("basic_rgb_planar")

    def test_copy_nv12(self):
        self._test_copy("basic_nv12")

    def test_copy_hevc10_p10(self):
        self._test_copy("hevc10_p10")

    def test_yuv_basic(self):
        self._test_yuv("basic")

    def test_yuv_basic_mpeg4(self):
        self._test_yuv("basic_mpeg4")

    def test_yuv_res_change(self):
        self._test_yuv("res_change")
    
    '''
    p10 decoding does not work and is not tested in other unittest (eg test_PyDecoder)
    def test_yuv_basic_hevc10(self):
        self._test_yuv("hevc10")
    '''

if __name__ == "__main__":
    unittest.main()
