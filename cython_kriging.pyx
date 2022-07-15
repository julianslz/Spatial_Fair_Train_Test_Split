cimport numpy
cimport cython
import numpy
import scipy
from libc.math cimport sqrt, fmax, exp


cdef inline double covariance(
        double x1,
        double y1,
        double x2,
        double y2,
        double decay
        ):
    cdef double dx = x2 - x1
    cdef double dy = y2 - y1
    cdef double distance = sqrt(dx * dx + dy * dy)

    return exp(-distance / decay)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_test_set_variances(
        numpy.ndarray[numpy.float64_t, ndim=1] x_train,
        numpy.ndarray[numpy.float64_t, ndim=1] y_train,
        numpy.ndarray[numpy.float64_t, ndim=1] x_test,
        numpy.ndarray[numpy.float64_t, ndim=1] y_test,
        ):

    cdef int n_train = len(x_train)
    cdef int n_test = len(x_test)

    cdef double decay = 500

    cdef int k, i, j
    cdef numpy.ndarray[numpy.float64_t, ndim=2] covariance_matrix = numpy.empty([n_train, n_train])
    cdef numpy.ndarray[numpy.float64_t, ndim=1] right_hand_side = numpy.empty(n_train)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] weights = numpy.empty(n_train)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] kriging_variance = numpy.empty(n_test)
    
    # Create matrix of covariances
    for i in range(n_train):
        for j in range(i + 1):
            
            # Fill the lower diagonal entries
            covariance_matrix[i, j] = covariance(
                    x_train[i],
                    y_train[i],
                    x_train[j],
                    y_train[j],
                    decay,
                )
            
            # Fill the upper diagonal too
            if j < i:
                covariance_matrix[j, i] = covariance_matrix[i, j]
                

    # Factor the matrix
    cholesky_factor, low = scipy.linalg.cho_factor(covariance_matrix)

    # Make and solve the kriging matrix, calculate the kriging estimate and variance
    for k in range(n_test):

        # Build the right hand side
        for i in range(n_train):
            right_hand_side[i] = covariance(
                x_train[i],
                y_train[i],
                x_test[k],
                y_test[k],
                decay,
            )

        weights = scipy.linalg.cho_solve((cholesky_factor, low), right_hand_side)
        kriging_variance[k] = covariance_matrix[0, 0] - (weights * right_hand_side).sum()
        
    return kriging_variance



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_variances_loo(numpy.ndarray[numpy.float64_t, ndim=1] x_train,
                      numpy.ndarray[numpy.float64_t, ndim=1] y_train):
    """
    Leave one out simple kriging variances.
    """

    cdef double decay = 500
    cdef int i, j
    cdef int ndata = len(x_train)
    
    cdef numpy.ndarray[numpy.float64_t, ndim=2] C = numpy.empty([ndata, ndata])
    cdef numpy.ndarray[numpy.float64_t, ndim=2] C_inv
    cdef numpy.ndarray[numpy.uint8_t, ndim=1, cast=True] e;
    cdef numpy.ndarray[numpy.float64_t, ndim=1] kriging_variance = numpy.empty(ndata)
    
    # Create matrix of covariances
    for i in range(ndata):
        for j in range(i + 1):
            
            # Fill the lower diagonal entries
            C[i, j] = covariance(
                    x_train[i],
                    y_train[i],
                    x_train[j],
                    y_train[j],
                decay,
            )
            
            # Fill the upper diagonal too
            if j < i:
                C[j, i] = C[i, j]
                
    # Invert the matrix
    C_inv = numpy.linalg.inv(C)
    
    for i in range(ndata):
        
        e = numpy.ones(ndata, dtype=bool)
        e[i] = 0
        
        kriging_variance[i] = C[i, i] + ((C_inv[i, e]/ C_inv[i, i]) * C[i, e]).sum()
        
    return kriging_variance



# =============================================================================
# =============================================================================

cdef inline double cova2(
        double x1,
        double y1,
        double x2,
        double y2,
        double cc,
        double aa,
        double anis,
        double rotmat1,
        double rotmat2,
        double rotmat3,
        double rotmat4,
        double maxcov
        ):
    """
    Calculate the covariance associated with a variogram model specified by
    a nugget effect and nested variogram structures
    :param x1: 
    :param y1: 
    :param x2: 
    :param y2: 
    :param cc: 
    :param aa: 
    :param anis: 
    :param rotmat1: 
    :param rotmat2: 
    :param rotmat3: 
    :param rotmat4: 
    :param maxcov: 
    :return: 
    """

    cdef double epsilon = 0.000001
    cdef double cova2
    cdef double dx = x2 - x1
    cdef double dy = y2 - y1
    cdef double comparer = dx * dx + dy * dy

    # Non-zero distance, loop over all the structures
    # Compute the appropriate structural distance
    cdef double dx1 = dx * rotmat1 + dy * rotmat2
    cdef double dy1 = (dx * rotmat3 + dy * rotmat4) / anis
    cdef double h = sqrt(fmax(dx1 * dx1 + dy1 * dy1, 0.0))

    # Gaussian model
    # cdef double hh = -3.0 * (h * h) / (aa * aa)
    # cova2_ = cc * exp(hh)

    # Spherical model
    hr = h / aa
    if hr < 1.0:
        cova2_ = cc * (1.0 - hr * (1.5 - 0.5 * hr * hr))

    # compare if you are in the same location
    if comparer < epsilon:
        cova2_ = maxcov

    return cova2_

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def simple_krig_var_one_at_a_time(
        int ndata,
        double anis,
        double cc,
        double aa,
        double nug,
        numpy.ndarray[numpy.float64_t, ndim=1] x_train,
        numpy.ndarray[numpy.float64_t, ndim=1] y_train,
        double rotmat1,
        double rotmat2,
        double rotmat3,
        double rotmat4,
        double maxcov
        ):
    
    cdef int iest, i, j
    cdef double sill = nug + cc
    cdef numpy.ndarray[numpy.float64_t, ndim=2] C = numpy.empty([ndata, ndata])
    cdef numpy.ndarray[numpy.float64_t, ndim=2] C_inv
    cdef numpy.ndarray[numpy.uint8_t, ndim=1, cast=True] e;
    cdef numpy.ndarray[numpy.float64_t, ndim=1] kriging_variance = numpy.empty(ndata)
    
    # Create matrix of covariances
    for i in range(ndata):
        for j in range(i + 1):
            
            # Fill the lower diagonal entries
            C[i, j] = cova2(
                    x_train[i],
                    y_train[i],
                    x_train[j],
                    y_train[j],
                    cc,
                    aa,
                    anis,
                    rotmat1,
                    rotmat2,
                    rotmat3,
                    rotmat4,
                    maxcov
                )
            
            # Fill the upper diagonal too
            if j < i:
                C[j, i] = C[i, j]
                
    # Invert the matrix
    C_inv = numpy.linalg.inv(C)
    
    for i in range(ndata):
        
        e = numpy.ones(ndata, dtype=bool)
        e[i] = 0
        
        kriging_variance[i] = C[i, i] + ((C_inv[i, e]/ C_inv[i, i]) * C[i, e]).sum()
        
    return kriging_variance
        


    
    
    
                
                
            
    
    



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def simple_krig_var(
        int ndata,
        int nest,
        double anis,
        double cc,
        double aa,
        double nug,
        numpy.ndarray[numpy.float64_t, ndim=1] x_train,
        numpy.ndarray[numpy.float64_t, ndim=1] y_train,
        numpy.ndarray[numpy.float64_t, ndim=1] x_test,
        numpy.ndarray[numpy.float64_t, ndim=1] y_test,
        double rotmat1,
        double rotmat2,
        double rotmat3,
        double rotmat4,
        double maxcov
        ):
    """
    Compute the kriging variance only at the testing locations.
    :param ndata:
    :param nest:
    :param anis:
    :param cc:
    :param aa:
    :param nug:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param rotmat1:
    :param rotmat2:
    :param rotmat3:
    :param rotmat4:
    :param maxcov:
    :return:
    """

    cdef int iest, idata, jdata
    cdef double sill = nug + cc
    cdef numpy.ndarray[numpy.float64_t, ndim=2] a = numpy.zeros([ndata, ndata])
    cdef numpy.ndarray[numpy.float64_t, ndim=1] r = numpy.zeros(ndata)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] s = numpy.zeros(ndata)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] rr = numpy.zeros(ndata)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] kriging_variance = numpy.full(nest, sill)


    # Create the covariance matrix
    for idata in range(ndata):
        for jdata in range(ndata):
            a[idata, jdata] = cova2(
                    x_train[idata],
                    y_train[idata],
                    x_train[jdata],
                    y_train[jdata],
                    cc,
                    aa,
                    anis,
                    rotmat1,
                    rotmat2,
                    rotmat3,
                    rotmat4,
                    maxcov
                )

    cholesky_factor, low = scipy.linalg.cho_factor(a)

    # Make and solve the kriging matrix, calculate the kriging estimate and variance
    for iest in range(nest):
        for idata in range(ndata):

            r[idata] = cova2(
                x_train[idata],
                y_train[idata],
                x_test[iest],
                y_test[iest],
                cc,
                aa,
                anis,
                rotmat1,
                rotmat2,
                rotmat3,
                rotmat4,
                maxcov
            )
            rr[idata] = r[idata]


        # from scipy.linalg import cho_factor, cho_solve
        s = scipy.linalg.cho_solve((cholesky_factor, low), r)

        # s = ksol_numpy(ndata, a, r)
        for idata in range(0, ndata):
            kriging_variance[iest] = kriging_variance[iest] - s[idata] * rr[idata]
    return kriging_variance


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def simple_krig_var2(
        int ndata,
        int nest,
        double anis,
        double cc,
        double aa,
        double nug,
        numpy.ndarray[numpy.float64_t, ndim=1] x_train,
        numpy.ndarray[numpy.float64_t, ndim=1] y_train,
        numpy.ndarray[numpy.float64_t, ndim=1] x_test,
        numpy.ndarray[numpy.float64_t, ndim=1] y_test,
        double rotmat1,
        double rotmat2,
        double rotmat3,
        double rotmat4,
        double maxcov
        ):
    """
    Compute the kriging variance only at the testing locations.
    :param ndata:
    :param nest:
    :param anis:
    :param cc:
    :param aa:
    :param nug:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param rotmat1:
    :param rotmat2:
    :param rotmat3:
    :param rotmat4:
    :param maxcov:
    :return:
    """

    cdef int iest, i, j
    cdef double sill = nug + cc
    cdef numpy.ndarray[numpy.float64_t, ndim=2] C = numpy.empty([ndata, ndata])
    cdef numpy.ndarray[numpy.float64_t, ndim=1] r = numpy.zeros(ndata)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] s = numpy.zeros(ndata)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] kriging_variance = numpy.full(nest, sill)
    
    # Create matrix of covariances
    for i in range(ndata):
        for j in range(i + 1):
            
            # Fill the lower diagonal entries
            C[i, j] = cova2(
                    x_train[i],
                    y_train[i],
                    x_train[j],
                    y_train[j],
                    cc,
                    aa,
                    anis,
                    rotmat1,
                    rotmat2,
                    rotmat3,
                    rotmat4,
                    maxcov
                )
            
            # Fill the upper diagonal too
            if j < i:
                C[j, i] = C[i, j]
                

    # Factor the matrix
    cholesky_factor, low = scipy.linalg.cho_factor(C)

    # Make and solve the kriging matrix, calculate the kriging estimate and variance
    for iest in range(nest):
        for i in range(ndata):

            r[i] = cova2(
                x_train[i],
                y_train[i],
                x_test[iest],
                y_test[iest],
                cc,
                aa,
                anis,
                rotmat1,
                rotmat2,
                rotmat3,
                rotmat4,
                maxcov
            )

        # from scipy.linalg import cho_factor, cho_solve
        s = scipy.linalg.cho_solve((cholesky_factor, low), r)
        
        kriging_variance[iest] = kriging_variance[iest] - (s * r).sum()
        
    return kriging_variance


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def solve_SK_equations(
        numpy.ndarray[numpy.float64_t, ndim=2] C,
        numpy.ndarray[numpy.float64_t, ndim=2] rhs,
        double sill
        ):
    """
    Compute the kriging variance only at the testing locations.
    :param ndata:
    :param nest:
    :param anis:
    :param cc:
    :param aa:
    :param nug:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param rotmat1:
    :param rotmat2:
    :param rotmat3:
    :param rotmat4:
    :param maxcov:
    :return:
    """
    
    cdef int n_data = C.shape[0]
    cdef int k_data = rhs.shape[1]
    assert C.shape[0] == C.shape[1]
    
    
    cdef numpy.ndarray[numpy.float64_t, ndim=1] kriging_variance = numpy.empty(k_data, sill)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] weights = numpy.empty(n_data)
    
    # Factor C = 
    cholesky_factor, low = scipy.linalg.cho_factor(C)
    
    for k in range(k_data):
        rhs_vector = rhs[:, k]
        weights = scipy.linalg.cho_solve((cholesky_factor, low), rhs_vector)
        
        kriging_variance[k] = kriging_variance[k] - (weights * rhs_vector).sum()
