/**
 * @author André Anjos <andre.anjos@idiap.ch>
 * @date Fri  4 Apr 15:20:24 2014 CEST
 *
 * @brief Helpers for color conversion
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "utils.h"

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

int check_and_allocate(Py_ssize_t input_dims, Py_ssize_t output_dims,
    boost::shared_ptr<PyBlitzArrayObject>& input,
    boost::shared_ptr<PyBlitzArrayObject>& output) {

  if (input->type_num != NPY_UINT8 &&
      input->type_num != NPY_UINT16 &&
      input->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "method only supports 8 or 16-bit unsigned integer or 64-bit floating-point arrays for input array `input', but you provided an array with data type `%s'", PyBlitzArray_TypenumAsString(input->type_num));
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
      PyErr_Format(PyExc_TypeError, "`output' array (`%s') should have a matching data type to the `input' array (`%s')", PyBlitzArray_TypenumAsString(output->type_num), PyBlitzArray_TypenumAsString(input->type_num));
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
