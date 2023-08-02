#from distutils.core import setup
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension
from Cython.Build import cythonize
#from Cython.Distutils import Extension
#from Cython.Distutils import build_ext
#from numpy.distutils.core import Extension as npExt
import cython_gsl




cy_modules = cythonize('SourceSubtraction/Tools/gsl_funcs.pyx')
e = cy_modules[0]

modules = [Extension('SourceSubtraction.Tools.gsl_funcs',
                     sources=e.sources,
                     libraries=cython_gsl.get_libraries(),
                     library_dirs=[cython_gsl.get_library_dir()],
                     include_dirs=[cython_gsl.get_cython_include_dir()],
                     extra_compile_args=['-fopenmp', '-Wl,-rpath,/usr/lib/x86_64-linux-gnu/'],
                     extra_link_args=['-fopenmp']),
           Extension('SourceSubtraction.Tools.pysla', 
                     sources=['SourceSubtraction/Tools/pysla.f90','SourceSubtraction/Tools/sla.f'],
                     f2py_options = [])
                   ]

setup(
    include_dirs = [cython_gsl.get_include()],
#    cmdclass = {'build_ext':build_ext},
    ext_modules = modules
)

print(cython_gsl.get_libraries(), cython_gsl.get_library_dir(), cython_gsl.get_include()) 
