cimport cython

from libc.math cimport sin, cos  # Import necessary C math functions (optional)

# (Optional) Define your complex integrand function (replace with your actual function)
cdef double complex f(double complex z):
    return sin(z.real) * cos(z.imag)

# Function to integrate using SciPy's quad function
cdef double complex integrate_complex_scipy(double a, double b):
    from scipy.integrate import quad  # Import SciPy's quad function within the Cython function

    # Call quad with the complex integrand function and limits
    cdef double complex result, error
    result, error = quad(func=f, a=a, b=b)  # SciPy returns both result and error

    # Handle errors if necessary (check error value or raise an exception)
    if error > 1e-5:
        raise ValueError("Integration error exceeded tolerance")

    return result
