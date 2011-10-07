#./waf-light --tools=compat15,swig,fc,compiler_fc,fc_config,fc_scan,gfortran,g95,ifort,gccdeps;


from textwrap import dedent

top = '.'
out = 'build'

def options(opt):
    opt.load('compiler_c')
    opt.load('compiler_fc')
    opt.load('python')
    opt.load('inplace', tooldir='tools')
    opt.add_option('--with-libpsht', help='path to libpsht to use for benchmark comparison '
                   '(NOTE: must be built with -fPIC)')
    opt.add_option('--no-openmp', action='store_true')

def configure(conf):
    conf.add_os_flags('PATH')
    conf.add_os_flags('PYTHON')
    conf.add_os_flags('PYTHONPATH')
    conf.add_os_flags('FWRAP')
    conf.add_os_flags('INCLUDES')
    conf.add_os_flags('LIB')
    conf.add_os_flags('LIBPATH')
    conf.add_os_flags('STLIB')
    conf.add_os_flags('STLIBPATH')
    conf.add_os_flags('FWRAPFLAGS')
    conf.add_os_flags('CFLAGS')
    conf.add_os_flags('LINKFLAGS')
    conf.add_os_flags('CYTHONFLAGS')
    conf.add_os_flags('CYTHON')

    conf.load('compiler_c')
    conf.load('compiler_fc')
    conf.check_fortran()
    conf.check_fortran_verbose_flag()
    conf.check_fortran_clib()

    conf.load('python')
    conf.check_python_version((2,5))
    conf.check_python_headers()

    conf.check_tool('numpy', tooldir='tools')
    conf.check_numpy_version(minver=(1,3))
    conf.check_tool('cython', tooldir='tools')
    conf.check_cython_version(minver=(0,11,1))
    conf.check_tool('inplace', tooldir='tools')

    # Libraries
    conf.check_libpsht()

    conf.env.CYTHONFLAGS = ['-a']

    if not conf.options.no_openmp:
        conf.env.CFLAGS_OPENMP = ['-fopenmp']
        conf.env.LINKFLAGS_OPENMP = ['-fopenmp']

def build(bld):
    #
    # Python wrappers
    #
    
    bld(source=(['psht4py/psht.pyx']),
        target='psht',
        use='NUMPY PSHT',
        features='c pyext cshlib')


from waflib.Configure import conf
from os.path import join as pjoin

@conf
def check_libpsht(conf):
    """
    Settings for libpsht
    """
    conf.start_msg("Checking for libpsht")
    prefix = conf.options.with_libpsht
    if not prefix:
        conf.fatal("--with-libpsht not used (FIXME)")
    conf.env.LIB_PSHT = ['psht', 'fftpack', 'c_utils']
    conf.env.LINKFLAGS_PSHT = ['-fopenmp']
    conf.env.LIBPATH_PSHT = [pjoin(prefix, 'lib')]
    conf.env.INCLUDES_PSHT = [pjoin(prefix, 'include')]
    # Check presence of libpsht in general
    cfrag = dedent('''\
    #include <psht.h>
    #include <psht_geomhelpers.h>
    psht_alm_info *x;
    psht_geom_info *y;
    pshtd_joblist *z;
    int main() {
    /* Only intended for compilation */
      psht_make_general_alm_info(10, 10, 1, NULL, 0, &x);
      psht_make_healpix_geom_info(4, 1, &y);
      pshtd_make_joblist(&z);
      pshtd_execute_jobs(x, y, z);
      return 0;
    }
    ''')
    conf.check_cc(
        fragment=cfrag,
        features = 'c',
        compile_filename='test.c',
        use='PSHT')
    conf.end_msg(prefix if prefix else True)


# vim:ft=python
