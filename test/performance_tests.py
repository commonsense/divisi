from csc.divisi.labeled_tensor import SparseLabeledTensor
import time
import os

def cpu():
    import resource
    return (resource.getrusage(resource.RUSAGE_SELF).ru_utime+
            resource.getrusage(resource.RUSAGE_SELF).ru_stime)

_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
    '''Private.
    '''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]

def memory(since=0.0):
    '''Return memory usage in bytes.
    '''
    return _VmB('VmSize:') - since

def resident(since=0.0):
    '''Return resident memory usage in bytes.
    '''
    return _VmB('VmRSS:') - since

def stacksize(since=0.0):
    '''Return stack size in bytes.
    '''
    return _VmB('VmStk:') - since

def performance_test(thunk, niter=100, print_result=True, test_name=""):
    start_memory = memory()
    start_time = time.time()
    start_cpu = cpu()

    # Just hold a reference to the thing that is returned so it isn't
    # garbage collected. This makes sense for most memory tests
    o = None
    for i in xrange(0, niter):
        o = thunk()

    total_time = time.time() - start_time
    total_cpu = cpu() - start_cpu
    total_memory = memory(start_memory)
    if print_result:
        print test_name, "(%d iterations)" % niter
        print "Total elapsed clock time:", total_time
        print "Per test clock time:", float(total_time) / niter
        print "Total elapsed CPU time:", total_cpu
        print "Per test CPU time:", float(total_cpu) / niter
        print "Memory usage (questionable accuracy due to gc):", total_memory
        print ""

    return total_time, total_cpu, total_memory
performance_test.__test__ = False

def insert_test(num_rows, num_cols):
    t = SparseLabeledTensor(ndim = 2)
    for i in xrange(0, num_rows):
        for j in xrange (0, num_cols):
            t[i, j] = 1
    return t
insert_test.__test__ = False

def iteration_test(iterable):
    for x in iterable:
        pass
iteration_test.__test__ = False


if __name__ == '__main__':
    performance_test(lambda: 0, niter=1, test_name="Null test")
    performance_test(lambda: insert_test(100, 100), niter=1, test_name="Tensor insert 100x100")
    performance_test(lambda: insert_test(1000, 1000), niter=1, test_name="Tensor insert 1000x1000")
    tensor = insert_test(100, 100)
    performance_test(lambda: tensor[0, :], niter=100, test_name="Tensor slice 100x100")
    performance_test(lambda: tensor.incremental_svd(k=4, niter=100), niter=1,test_name="Iterative SVD test, 100x100")
    tensor = insert_test(1000, 1000)
    performance_test(lambda: tensor[0, :], niter=10, test_name="Tensor slice 1000x1000")
    performance_test(lambda: iteration_test(tensor[0, :]), niter=10, test_name="Tensor (1000x1000) slice iteration test")
