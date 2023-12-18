from conan import ConanFile

class FfmpegRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"

    def requirements(self):
        ffmpeg = "ffmpeg/6.1"
        self.requires(ffmpeg)

    def configure(self):
        self.options["ffmpeg/*"].shared = True

        #self.options["ffmpeg/*"].with_xcb = False
        #self.options["ffmpeg/*"].with_lzma = False
        #self.options["ffmpeg/*"].with_bzip2 = False
        #self.options["ffmpeg/*"].with_pulse = False
        #self.options["ffmpeg/*"].with_vaapi = False
        #self.options["ffmpeg/*"].with_vdpau = False        
        #self.options["ffmpeg/*"].with_vulkan = False        
        #self.options["ffmpeg/*"].with_libx264 = False
        #self.options["ffmpeg/*"].with_libx265 = False
        #self.options["ffmpeg/*"].with_libalsa = False
        #self.options["ffmpeg/*"].with_libiconv = False
        #self.options["ffmpeg/*"].with_freetype = False
        #self.options["ffmpeg/*"].with_libmp3lame = False
        #self.options["ffmpeg/*"].with_libfdk_aac = False
        
        #self.options["ffmpeg/*"].with_programs = False
        #self.options["ffmpeg/*"].disable_all_encoders = True
        #self.options["ffmpeg/*"].disable_all_hardware_accelerators = True
