/**
 * @author André Anjos <andre.anjos@idiap.ch>
 * @date Thu  3 Apr 18:47:12 2014 CEST
 *
 * @brief Binds color converters to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include <xbob.blitz/cppapi.h>
#include <xbob.blitz/cleanup.h>
#include <bob/ip/color.h>

static int check_2doutput_for_3dinput(
    boost::shared_ptr<PyBlitzArrayObject>& input,
    boost::shared_ptr<PyBlitzArrayObject>& output) {

  if (output->shape[0] != input->shape[1]) {
    PyErr_Format(PyExc_RuntimeError, "2D `output' array should have %" PY_FORMAT_SIZE_T "d rows (1st dimension extent) matching the number of rows (2nd dimension extent) in 3D `input', not %" PY_FORMAT_SIZE_T "d planes", input->shape[1], output->shape[0]);
    return 0;
  }

  if (output->shape[1] != input->shape[2]) {
    PyErr_Format(PyExc_RuntimeError, "2D `output' array should have %" PY_FORMAT_SIZE_T "d columns (2nd dimension extent) matching the number of columns (3rd dimension extent) in 3D `input', not %" PY_FORMAT_SIZE_T "d rows", input->shape[2], output->shape[1]);
    return 0;
  }

  return 1;
}

static int check_3doutput_for_2dinput(
    boost::shared_ptr<PyBlitzArrayObject>& input,
    boost::shared_ptr<PyBlitzArrayObject>& output) {

  if (output->shape[0] != 3) {
    PyErr_Format(PyExc_RuntimeError, "3D `output' array should have 3 planes (1st dimension extent) matching the number of planes in colored images, not %" PY_FORMAT_SIZE_T "d planes", output->shape[0]);
    return 0;
  }

  if (output->shape[1] != input->shape[0]) {
    PyErr_Format(PyExc_RuntimeError, "3D `output' array should have %" PY_FORMAT_SIZE_T "d rows (2nd dimension extent) matching the number of rows (1st dimension extent) in 2D `input', not %" PY_FORMAT_SIZE_T "d planes", input->shape[0], output->shape[1]);
    return 0;
  }

  if (output->shape[2] != input->shape[1]) {
    PyErr_Format(PyExc_RuntimeError, "3D `output' array should have %" PY_FORMAT_SIZE_T "d columns (3rd dimension extent) matching the number of columns (2nd dimension extent) in 2D `input', not %" PY_FORMAT_SIZE_T "d rows", input->shape[1], output->shape[2]);
    return 0;
  }

  return 1;
}

static int check_3doutput_for_3dinput(
    boost::shared_ptr<PyBlitzArrayObject>& input,
    boost::shared_ptr<PyBlitzArrayObject>& output) {

  if (output->shape[0] != 3) {
    PyErr_Format(PyExc_RuntimeError, "3D `output' array should have 3 planes (1st dimension extent) matching the number of planes in colored images, not %" PY_FORMAT_SIZE_T "d planes", output->shape[0]);
    return 0;
  }

  if (output->shape[1] != input->shape[1]) {
    PyErr_Format(PyExc_RuntimeError, "3D `output' array should have %" PY_FORMAT_SIZE_T "d rows (2nd dimension extent) matching the number of rows (2nd dimension extent) in 3D `input', not %" PY_FORMAT_SIZE_T "d planes", input->shape[1], output->shape[1]);
    return 0;
  }

  if (output->shape[2] != input->shape[2]) {
    PyErr_Format(PyExc_RuntimeError, "3D `output' array should have %" PY_FORMAT_SIZE_T "d columns (3rd dimension extent) matching the number of columns (3rd dimension extent) in 3D `input', not %" PY_FORMAT_SIZE_T "d rows", input->shape[2], output->shape[2]);
    return 0;
  }

  return 1;
}

static int check_and_allocate(Py_ssize_t input_dims, Py_ssize_t output_dims,
    boost::shared_ptr<PyBlitzArrayObject>& input,
    boost::shared_ptr<PyBlitzArrayObject>& output) {

  if (input->type_num != NPY_UINT8 &&
      input->type_num != NPY_UINT16 &&
      input->type_num != NPY_FLOAT64) {
    PyErr_SetString(PyExc_TypeError, "method only supports 8 or 16-bit unsigned integer or 64-bit floating-point arrays for input array `input', but you provided an array with data type `%s'", PyBlitzArray_TypenumAsString(input->type_num));
    return 0;
  }

  if (input_dims != input->ndim) {
    PyErr_Format(PyExc_TypeError, "method only accepts %" PY_FORMAT_SIZE_T "d-dimensional arrays as `input', not %" PY_FORMAT_SIZE_T "dD arrays", input_dims, input->ndim);
    return 0;
  }

  if (input_dims == 3 && (input->shape[0] != 3)) {
    PyErr_Format(PyExc_TypeError, "method only accepts 3-dimensional arrays with shape (3, height, width), not (%" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d)", input->shape[0], input->shape[1], input->shape[2]);
    return 0;
  }

  if (output) {

    if (output->type_num != input->type_num) {
      PyErr_SetString(PyExc_TypeError, "`output' array (`%s') should have a matching data type to the `input' array (`%s')", PyBlitzArray_TypenumAsString(output->type_num), PyBlitzArray_TypenumAsString(input->type_num));
      return 0;
    }

    if (output_dims != output->ndim) {
      PyErr_Format(PyExc_TypeError, "method only accepts %" PY_FORMAT_SIZE_T "d-dimensional arrays as `output', not %" PY_FORMAT_SIZE_T "dD arrays", output_dims, output->ndim);
      return 0;
    }

    //check shape
    if (input_dims == 3 && output_dims == 3) { //e.g. RGB -> HSV
      if (!check_3doutput_for_3dinput(input, output)) return 0;
    }
    else if (input_dims == 3 && output_dims == 2) { //e.g. RGB -> Gray
      if (!check_2doutput_for_3dinput(input, output)) return 0;
    }
    else if (input_dims == 2 && output_dims == 3) { //e.g. Gray -> RGB
      if (!check_3doutput_for_2dinput(input, output)) return 0;
    }
    else {
      PyErr_Format(PyExc_NotImplementedError, "cannot check for %" PY_FORMAT_SIZE_T "dD input and %" PY_FORMAT_SIZE_T "dD output - DEBUG ME", input_dims, output_dims);
      return 0;
    }
  }

  else {

    //allocate
    Py_ssize_t shape[3];
    if (input_dims == 3 && output_dims == 3) {
      shape[0] = input->shape[0];
      shape[1] = input->shape[1];
      shape[2] = input->shape[2];
    }
    else if (input_dims == 3 && output_dims == 2) { //e.g. RGB -> Gray
      shape[0] = input->shape[1];
      shape[1] = input->shape[2];
      shape[2] = 0;
    }
    else if (input_dims == 2 && output_dims == 3) { //e.g. Gray -> RGB
      shape[0] = 3;
      shape[1] = input->shape[0];
      shape[2] = input->shape[1];
    }
    else {
      PyErr_Format(PyExc_NotImplementedError, "cannot allocate for %" PY_FORMAT_SIZE_T "dD input and %" PY_FORMAT_SIZE_T "dD output - DEBUG ME", input_dims, output_dims);
      return 0;
    }

    auto tmp = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(input->type_num,
        output_dims, shape);
    if (!tmp) return 0;
    output = make_safe(tmp);

  }

  return 1;

}

static PyObject* PyBobIpColor_RgbToGray_Array
  (PyObject* self, PyObject* args, PyObject* kwds) {

  static const char* const_kwlist[] = {"input", "output", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* input = 0;
  PyBlitzArrayObject* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&", kwlist,
        &PyBlitzArray_Converter, &input,
        &PyBlitzArray_OutputConverter, &output
        )) return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);
  auto output_ = make_xsafe(output);

  check_and_allocate(3, input_, output_);

  output = output_.get();

  switch (input->type_num) {
    case NPY_UINT8:
      bob::ip::rgb_to_gray(
          *PyBlitzArrayCxx_AsBlitz<uint8_t,3>(input),
          *PyBlitzArrayCxx_AsBlitz<uint8_t,2>(output),
          );
    case NPY_UINT16:
      bob::ip::rgb_to_gray(
          *PyBlitzArrayCxx_AsBlitz<uint16_t,3>(input),
          *PyBlitzArrayCxx_AsBlitz<uint16_t,2>(output),
          );
    case NPY_FLOAT64:
      bob::ip::rgb_to_gray(
          *PyBlitzArrayCxx_AsBlitz<double,3>(input),
          *PyBlitzArrayCxx_AsBlitz<double,2>(output),
          );
    default:
      PyErr_Format(PyExc_NotImplementedError, "function has no support for data type `%s', choose from uint8, uint16 or float64", PyBlitzArray_TypenumAsString(input->type_num));
      return 0;
  }

  Py_INCREF(output);
  return PyBlitzArray_NUMPY_WRAP(reinterpret_cast<PyObject*>(output));

}

static PyObject* PyBobIpColor_RgbToGray_Scalar
  (PyObject* self, PyObject* args, PyObject* kwds) {

  static const char* const_kwlist[] = {"r", "g", "b", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* r = 0;
  PyObject* g = 0;
  PyObject* b = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO", kwlist, &r, &g, &b)) return 0;

  //checks all input objects are scalars
  if (!PyArray_IsAnyScalar(r)) {
    PyErr_Format(PyExc_TypeError, "input element `r' should be a python or numpy scalar, not `%s'", Py_TYPE(r)->tp_name);
    return 0;
  }

  if (!PyArray_IsAnyScalar(g)) {
    PyErr_Format(PyExc_TypeError, "input element `g' should be a python or numpy scalar, not `%s'", Py_TYPE(g)->tp_name);
    return 0;
  }

  if (!PyArray_IsAnyScalar(b)) {
    PyErr_Format(PyExc_TypeError, "input element `b' should be a python or numpy scalar, not `%s'", Py_TYPE(b)->tp_name);
    return 0;
  }

  //checks all scalars are of the same type
  if (Py_TYPE(r) != Py_TYPE(g)) {
    PyErr_Format(PyExc_TypeError, "input scalar type for `g' (`%s') differs from the type for element `r' (`%s')", Py_TYPE(g)->tp_name, Py_TYPE(r)->tp_name);
    return 0;
  }

  if (Py_TYPE(r) != Py_TYPE(b)) {
    PyErr_Format(PyExc_TypeError, "input scalar type for `b' (`%s') differs from the type for element `r' and `g' (`%s')", Py_TYPE(b)->tp_name, Py_TYPE(r)->tp_name);
    return 0;
  }

  //checks the type for one of the channels, cast all
  int type_num = PyArray_ObjectType(r, NPY_NOTYPE);

  switch (input->type_num) {
    case NPY_UINT8:
      {
        uint8_t retval;
        bob::ip::rgb_to_gray_one(
            PyBlitzArrayCxx_AsCScalar<uint8_t>(r),
            PyBlitzArrayCxx_AsCScalar<uint8_t>(g),
            PyBlitzArrayCxx_AsCScalar<uint8_t>(b),
            retval
            );
        return PyBlitzArrayCxx_FromCScalar(retval);
      }
    case NPY_UINT16:
      {
        uint16_t retval;
        bob::ip::rgb_to_gray_one(
            PyBlitzArrayCxx_AsCScalar<uint16_t>(r),
            PyBlitzArrayCxx_AsCScalar<uint16_t>(g),
            PyBlitzArrayCxx_AsCScalar<uint16_t>(b),
            retval
            );
        return PyBlitzArrayCxx_FromCScalar(retval);
      }
    case NPY_FLOAT64:
      {
        double retval;
        bob::ip::rgb_to_gray_one(
            PyBlitzArrayCxx_AsCScalar<double>(r),
            PyBlitzArrayCxx_AsCScalar<double>(g),
            PyBlitzArrayCxx_AsCScalar<double>(b),
            retval
            );
        return PyBlitzArrayCxx_FromCScalar(retval);
      }
    default:
      PyErr_Format(PyExc_NotImplementedError, "function has no support for data type `%s', choose from uint8, uint16 or float64", Py_TYPE(self)->tp_name, PyBlitzArray_TypenumAsString(input->type_num));
  }

  return 0;
}

PyObject* PyBobIpColor_RgbToGray (PyObject*, PyObject* args, PyObject* kwds) {

  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  switch (nargs) {

    case 1: //should pass an array
    case 2:
      return PyBobIpColor_RgbToGray_Array(args, kwds);

    case 3:
      return PyBobIpColor_RgbToGray_Scalar(args, kwds);

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - function requires 1, 2 or 3 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", nargs);

  }

  return 0;

}

/**
static object gray_to_rgb(const object& y, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t r, g, b;
        bob::ip::gray_to_rgb_one((uint8_t)extract<uint8_t>(y), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t r, g, b;
        bob::ip::gray_to_rgb_one((uint16_t)extract<uint8_t>(y), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_float64:
      {
        double r, g, b;
        bob::ip::gray_to_rgb_one((double)extract<uint8_t>(y), r, g, b);
        return make_tuple(r, g, b);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}


// RGB to HSV and vice versa
static object rgb_to_hsv(const object& r, const object& g, const object& b, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t h, s, v;
        bob::ip::rgb_to_hsv_one((uint8_t)extract<uint8_t>(r), (uint8_t)extract<uint8_t>(g), (uint8_t)extract<uint8_t>(b), h, s, v);
        return make_tuple(h, s, v);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t h, s, v;
        bob::ip::rgb_to_hsv_one((uint16_t)extract<uint16_t>(r), (uint16_t)extract<uint16_t>(g), (uint16_t)extract<uint16_t>(b), h, s, v);
        return make_tuple(h, s, v);
      }
    case bob::core::array::t_float64:
      {
        double h, s, v;
        bob::ip::rgb_to_hsv_one((double)extract<double>(r), (double)extract<double>(g), (double)extract<double>(b), h, s, v);
        return make_tuple(h, s, v);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}

static object hsv_to_rgb(const object& h, const object& s, const object& v, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t r, g, b;
        bob::ip::hsv_to_rgb_one((uint8_t)extract<uint8_t>(h), (uint8_t)extract<uint8_t>(s), (uint8_t)extract<uint8_t>(v), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t r, g, b;
        bob::ip::hsv_to_rgb_one((uint16_t)extract<uint16_t>(h), (uint16_t)extract<uint16_t>(s), (uint16_t)extract<uint16_t>(v), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_float64:
      {
        double r, g, b;
        bob::ip::hsv_to_rgb_one((double)extract<double>(h), (double)extract<double>(s), (double)extract<double>(v), r, g, b);
        return make_tuple(r, g, b);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}


// RGB to HSL and vice versa
static object rgb_to_hsl(const object& r, const object& g, const object& b, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t h, s, l;
        bob::ip::rgb_to_hsl_one((uint8_t)extract<uint8_t>(r), (uint8_t)extract<uint8_t>(g), (uint8_t)extract<uint8_t>(b), h, s, l);
        return make_tuple(h, s, l);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t h, s, l;
        bob::ip::rgb_to_hsl_one((uint16_t)extract<uint16_t>(r), (uint16_t)extract<uint16_t>(g), (uint16_t)extract<uint16_t>(b), h, s, l);
        return make_tuple(h, s, l);
      }
    case bob::core::array::t_float64:
      {
        double h, s, l;
        bob::ip::rgb_to_hsl_one((double)extract<double>(r), (double)extract<double>(g), (double)extract<double>(b), h, s, l);
        return make_tuple(h, s, l);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}

static object hsl_to_rgb(const object& h, const object& s, const object& l, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t r, g, b;
        bob::ip::hsl_to_rgb_one((uint8_t)extract<uint8_t>(h), (uint8_t)extract<uint8_t>(s), (uint8_t)extract<uint8_t>(l), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t r, g, b;
        bob::ip::hsl_to_rgb_one((uint16_t)extract<uint16_t>(h), (uint16_t)extract<uint16_t>(s), (uint16_t)extract<uint16_t>(l), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_float64:
      {
        double r, g, b;
        bob::ip::hsl_to_rgb_one((double)extract<double>(h), (double)extract<double>(s), (double)extract<double>(l), r, g, b);
        return make_tuple(r, g, b);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}


// RGB to YUV and vice versa
static object rgb_to_yuv(const object& r, const object& g, const object& b, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t y, u, v;
        bob::ip::rgb_to_yuv_one((uint8_t)extract<uint8_t>(r), (uint8_t)extract<uint8_t>(g), (uint8_t)extract<uint8_t>(b), y, u, v);
        return make_tuple(y, u, v);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t y, u, v;
        bob::ip::rgb_to_yuv_one((uint16_t)extract<uint16_t>(r), (uint16_t)extract<uint16_t>(g), (uint16_t)extract<uint16_t>(b), y, u, v);
        return make_tuple(y, u, v);
      }
    case bob::core::array::t_float64:
      {
        double y, u, v;
        bob::ip::rgb_to_yuv_one((double)extract<double>(r), (double)extract<double>(g), (double)extract<double>(b), y, u, v);
        return make_tuple(y, u, v);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}

static object yuv_to_rgb(const object& y, const object& u, const object& v, const object& dtype){
  bob::python::dtype d(dtype);
  switch (d.eltype()) {
    case bob::core::array::t_uint8:
      {
        uint8_t r, g, b;
        bob::ip::yuv_to_rgb_one((uint8_t)extract<uint8_t>(y), (uint8_t)extract<uint8_t>(u), (uint8_t)extract<uint8_t>(v), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_uint16:
      {
        uint16_t r, g, b;
        bob::ip::yuv_to_rgb_one((uint16_t)extract<uint16_t>(y), (uint16_t)extract<uint16_t>(u), (uint16_t)extract<uint16_t>(v), r, g, b);
        return make_tuple(r, g, b);
      }
    case bob::core::array::t_float64:
      {
        double r, g, b;
        bob::ip::yuv_to_rgb_one((double)extract<double>(y), (double)extract<double>(u), (double)extract<double>(v), r, g, b);
        return make_tuple(r, g, b);
      }
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator is not supported for date type '%s'",
        d.cxx_str().c_str());
  }
}


//a few methods to return a dynamically allocated converted object
static
void py_rgb_to_hsv (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::rgb_to_hsv(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::rgb_to_hsv(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::rgb_to_hsv(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_rgb_to_hsv2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info);
  py_rgb_to_hsv(from, to);
  return to.self();
}

static
void py_hsv_to_rgb (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::hsv_to_rgb(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::hsv_to_rgb(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::hsv_to_rgb(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_hsv_to_rgb2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info);
  py_hsv_to_rgb(from, to);
  return to.self();
}

static
void py_rgb_to_hsl (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::rgb_to_hsl(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::rgb_to_hsl(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::rgb_to_hsl(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_rgb_to_hsl2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info);
  py_rgb_to_hsl(from, to);
  return to.self();
}

static
void py_hsl_to_rgb (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::hsl_to_rgb(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::hsl_to_rgb(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::hsl_to_rgb(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_hsl_to_rgb2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info);
  py_hsl_to_rgb(from, to);
  return to.self();
}

static
void py_rgb_to_yuv (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::rgb_to_yuv(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::rgb_to_yuv(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::rgb_to_yuv(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_rgb_to_yuv2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info);
  py_rgb_to_yuv(from, to);
  return to.self();
}

static
void py_yuv_to_rgb (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::yuv_to_rgb(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::yuv_to_rgb(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::yuv_to_rgb(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_yuv_to_rgb2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info);
  py_yuv_to_rgb(from, to);
  return to.self();
}

static
void py_rgb_to_gray (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,2> to_ = to.bz<uint8_t,2>();
        bob::ip::rgb_to_gray(from.bz<uint8_t,3>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,2> to_ = to.bz<uint16_t,2>();
        bob::ip::rgb_to_gray(from.bz<uint16_t,3>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,2> to_ = to.bz<double,2>();
        bob::ip::rgb_to_gray(from.bz<double,3>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_rgb_to_gray2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  if (info.nd != 3) {
    PYTHON_ERROR(TypeError,
      "input type must have at least 3 dimensions, but you gave me '%s'",
      info.str().c_str());
  }
  bob::python::ndarray to(info.dtype, info.shape[1], info.shape[2]);
  py_rgb_to_gray(from, to);
  return to.self();
}


static
void py_gray_to_rgb (bob::python::const_ndarray from, bob::python::ndarray to)
{
  switch (from.type().dtype) {
    case bob::core::array::t_uint8:
      {
        blitz::Array<uint8_t,3> to_ = to.bz<uint8_t,3>();
        bob::ip::gray_to_rgb(from.bz<uint8_t,2>(), to_);
      }
      break;
    case bob::core::array::t_uint16:
      {
        blitz::Array<uint16_t,3> to_ = to.bz<uint16_t,3>();
        bob::ip::gray_to_rgb(from.bz<uint16_t,2>(), to_);
      }
      break;
    case bob::core::array::t_float64:
      {
        blitz::Array<double,3> to_ = to.bz<double,3>();
        bob::ip::gray_to_rgb(from.bz<double,2>(), to_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError,
        "color conversion operator does not support array with type '%s'",
        from.type().str().c_str());
  }
}

static object py_gray_to_rgb2 (bob::python::const_ndarray from) {
  const bob::core::array::typeinfo& info = from.type();
  bob::python::ndarray to(info.dtype, (size_t)3, info.shape[0], info.shape[1]);
  py_gray_to_rgb(from, to);
  return to.self();
}


static const char* rgb_to_hsv_doc = "Takes a 3-dimensional array encoded as RGB and sets the second array with HSV equivalents as determined by rgb_to_hsv_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
static const char* hsv_to_rgb_doc = "Takes a 3-dimensional array encoded as HSV and sets the second array with RGB equivalents as determined by hsv_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
static const char* rgb_to_hsl_doc = "Takes a 3-dimensional array encoded as RGB and sets the second array with HSL equivalents as determined by rgb_to_hsl_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
static const char* hsl_to_rgb_doc = "Takes a 3-dimensional array encoded as HSL and sets the second array with RGB equivalents as determined by hsl_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
static const char* rgb_to_yuv_doc = "Takes a 3-dimensional array encoded as RGB and sets the second array with YUV (Y'CbCr) equivalents as determined by rgb_to_yuv_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
static const char* yuv_to_rgb_doc = "Takes a 3-dimensional array encoded as YUV (Y'CbCr) and sets the second array with RGB equivalents as determined by yuv_to_rgb_one(). The array must be organized in such a way that the color bands are represented by the first dimension.  Its shape should be something like (3, width, height) or (3, height, width). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported.";
static const char* rgb_to_gray_doc = "Takes a 3-dimensional array encoded as RGB and sets the second array with gray equivalents as determined by rgb_to_gray_one(). The array must be organized in such a way that the color bands are represented by the first dimension. Its shape should be something like (3, width, height) or (3, height, width). The output array is a 2D array with the same element type. The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported";
static const char* gray_to_rgb_doc = "Takes a 2-dimensional array encoded as grays and sets the second array with RGB equivalents as determined by gray_to_rgb_one(). The output array has to have the required size for the conversion otherwise an exception is raised (except for versions allocating the returned arrays). WARNING: As of this time only C-style storage arrays are supported";

void bind_ip_color()
{
  // Single pixel conversions
  def("rgb_to_hsv", &rgb_to_hsv, (arg("red"), arg("green"), arg("blue"), arg("dtype")), "Converts a RGB color-pixel to HSV as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,v) values.\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("hsv_to_rgb", &hsv_to_rgb, (arg("hue"), arg("saturation"), arg("value"), arg("dtype")), "Converts a HSV color-pixel to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("rgb_to_hsl", &rgb_to_hsl, (arg("red"), arg("green"), arg("blue"), arg("dtype")), "Converts a RGB color-pixel to HSL as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (h,s,l) values.\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("hsl_to_rgb", &hsl_to_rgb, (arg("hue"), arg("saturation"), arg("lightness"), arg("dtype")), "Converts a HSL color-pixel to RGB as defined in http://en.wikipedia.org/wiki/HSL_and_HSV. Returns a tuple with (r,g,b) values.\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("rgb_to_yuv", &rgb_to_yuv, (arg("red"), arg("green"), arg("blue"), arg("dtype")), "Converts a RGB color-coded pixel to YUV (Y'CbCr) using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (y,u,v) values.\n\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("yuv_to_rgb", &yuv_to_rgb, (arg("y"), arg("u"), arg("v"), arg("dtype")), "Converts a YUV (Y'CbCr) color-coded pixel to RGB using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://en.wikipedia.org/wiki/YCbCr and here: http://www.fourcc.org/fccyvrgb.php. Returns a tuple with (r,g,b) values.\n\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("rgb_to_gray", &rgb_to_gray, (arg("red"), arg("green"), arg("blue"), arg("dtype")), "Converts a RGB color-coded pixel to Grayscale using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed here: http://www.fourcc.org/fccyvrgb.php. Returns only the gray value (Y component) in the desired data format. This method is more efficient than calling rgb_to_yuv() method just to extract the Y component.\n\n Depending on the dtype parameter, different types of data is expected:\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");
  def("gray_to_rgb", &gray_to_rgb, (arg("y"), arg("dtype")), "Converts a grayscale pixel to RGB by copying the gray value to all 3 bands. Returns a tuple with (r,g,b) values. This method is just here for convenience.\n Depending on the dtype parameter, different types of data is expected:\n\n - 'float': float values between 0 and 1\n - 'uint8': integers between 0 and 255\n - 'uint16': integers between 0 and 65535");

  // image conversions from source to target image
  def("rgb_to_hsv", &py_rgb_to_hsv, (arg("rgb"), arg("hsv")), rgb_to_hsv_doc);
  def("hsv_to_rgb", &py_hsv_to_rgb, (arg("hsv"), arg("rgb")), hsv_to_rgb_doc);
  def("rgb_to_hsl", &py_rgb_to_hsl, (arg("rgb"), arg("hsl")), rgb_to_hsl_doc);
  def("hsl_to_rgb", &py_hsl_to_rgb, (arg("hsl"), arg("rgb")), hsl_to_rgb_doc);
  def("rgb_to_yuv", &py_rgb_to_yuv, (arg("rgb"), arg("yuv")), rgb_to_yuv_doc);
  def("yuv_to_rgb", &py_yuv_to_rgb, (arg("yuv"), arg("rgb")), yuv_to_rgb_doc);
  def("rgb_to_gray", &py_rgb_to_gray, (arg("rgb"), arg("gray")), rgb_to_gray_doc);
  def("gray_to_rgb", &py_gray_to_rgb, (arg("gray"), arg("rgb")), gray_to_rgb_doc);

  // more pythonic versions that return a dynamically allocated result
  def("rgb_to_hsv", &py_rgb_to_hsv2, (arg("rgb")), rgb_to_hsv_doc);
  def("hsv_to_rgb", &py_hsv_to_rgb2, (arg("hsv")), hsv_to_rgb_doc);
  def("rgb_to_hsl", &py_rgb_to_hsl2, (arg("rgb")), rgb_to_hsl_doc);
  def("hsl_to_rgb", &py_hsl_to_rgb2, (arg("hsl")), hsl_to_rgb_doc);
  def("rgb_to_yuv", &py_rgb_to_yuv2, (arg("rgb")), rgb_to_yuv_doc);
  def("yuv_to_rgb", &py_yuv_to_rgb2, (arg("yuv")), yuv_to_rgb_doc);
  def("rgb_to_gray", &py_rgb_to_gray2, (arg("rgb")), rgb_to_gray_doc);
  def("gray_to_rgb", &py_gray_to_rgb2, (arg("gray")), gray_to_rgb_doc);
}

**/
