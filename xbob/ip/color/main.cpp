/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri  4 Apr 15:02:59 2014 CEST
 *
 * @brief Bindings to bob::ip color converters
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <xbob.blitz/capi.h>
#include <xbob.blitz/cleanup.h>
#include <xbob.extension/documentation.h>

extern PyObject* PyBobIpColor_RgbToGray (PyObject*, PyObject*, PyObject*);
static xbob::extension::FunctionDoc s_rgb_to_gray = xbob::extension::FunctionDoc(
    "rgb_to_gray",

    "Converts an RGB color-coded pixel or a full array (image) to grayscale",

    "This function converts an RGB color-coded pixel or a full RGB array to "
    "grayscale using the CCIR 601 (Kb = 0.114, Kr = 0.299) norm as discussed "
    "`on this website <http://www.fourcc.org/fccyvrgb.php>`_. It returns only "
    "the gray value (Y component) in the desired data format. This method is "
    "more efficient than calling rgb_to_yuv() method just to extract the Y "
    "component.\n"
    "\n"
    "The input is expected to be either an array or a scalar. If you input an "
    "array, it is expected to assume the shape ``(3, height, width)``, "
    "representing an image encoded in RGB, in this order, with the specified "
    "``height`` and ``width``; or a set of 3 scalars defining the input R, G "
    "and B in a discrete way. The output array may be optionally provided. "
    "In such a case, it should be a 2D array with the same number of columns "
    "and rows as the input, and have have the same data type. If an output "
    "array is not provided, one will be allocated internally. In any case, "
    "the output array is always returned.\n"
    "\n"
    "If the input is of scalar type, this method will return the gray-scaled "
    "version for a pixel with the 3 discrete values for red, green and blue. "
    "\n"
    "\n"
    ".. note::\n"
    "\n"
    "   If you provide python scalars, then you should provide 3 values that "
    "share the same scalar type. Type mixing will raise a "
    ":py:class:`TypeError` exception.\n"
    "\n"
    ".. note::\n"
    "\n"
    "   This method only supports arrays and scalars of the following data "
    "types:\n"
    "\n"
    "   * :py:class:`numpy.uint8`\n"
    "   * :py:class:`numpy.uint16`\n"
    "   * :py:class:`numpy.float64` (or the native python ``float``)\n"
    "   \n"
    "   To create an object with a scalar type that will be accepted by this "
    "   method, use a construction like the following:\n"
    "   \n"
    "   .. code-block:: python\n"
    "      \n"
    "      >> import numpy\n"
    "      >> r = numpy.uint8(32)\n"
    "\n"
    )
    .add_prototype("input, output", "array")
    .add_prototype("r, g, b", "y")
    .add_parameter("input", "array_like (uint8|uint16|float64, 3D)", "Input array containing an image with the shape ``(3, height, width)``")
    .add_parameter("output", "array (uint8|uint16|float64, 2D), optional", "Output array - if provided, should have matching data type to ``input``. The shape should be ``(height, width)``")
    .add_parameter("r, g, b", "scalar (uint8|uint16|float64)", "Discrete pixel values for the red, green and blue channels")
    .add_return("array", "array_like (uint8|uint16|float64, 2D)", "The ``output`` array is returned by the function. If one was not provided, a new one is allocated internally")
    .add_return("y", "scalar (uint8|uint16|float64)", "A scalar is returned when this function is fed discrete RGB values. The type matches the input pixel values")
;

extern PyObject* PyBobIpColor_GrayToRgb (PyObject*, PyObject*, PyObject*);
static xbob::extension::FunctionDoc s_gray_to_rgb = xbob::extension::FunctionDoc(
    "gray_to_rgb",

    "Converts a gray pixel or a full array (image) to RGB",

    "This function converts a gray pixel or a gray array representing an "
    "image to a monochrome colored equivalent. This method is implemented "
    "for completeness and is equivalent to replicating the Y pixel value "
    "over the three RGB bands\n"
    "\n"
    "The input is expected to be either an array or a scalar. If you input an "
    "array, it is expected to assume the shape ``(height, width)``, "
    "representing an image encoded in gray scale, with the specified "
    "``height`` and ``width``; or a single scalar defining the input for Y "
    "in a discrete way. The output array may be optionally provided. "
    "In such a case, it should be a 3D array with the same number of columns "
    "and rows as the input, and have have the same data type. The number of "
    "color planes (first dimension) of such array should be ``3``. If an "
    "output array is not provided, one will be allocated internally. In any "
    "case, the output array is always returned.\n"
    "\n"
    "If the input is of scalar type, this method will return a tuple with "
    "the 3 discrete values for red, green and blue.\n"
    "\n"
    ".. note::\n"
    "\n"
    "   This method only supports arrays and scalars of the following data "
    "types:\n"
    "\n"
    "   * :py:class:`numpy.uint8`\n"
    "   * :py:class:`numpy.uint16`\n"
    "   * :py:class:`numpy.float64` (or the native python ``float``)\n"
    "   \n"
    "   To create an object with a scalar type that will be accepted by this "
    "   method, use a construction like the following:\n"
    "   \n"
    "   .. code-block:: python\n"
    "      \n"
    "      >> import numpy\n"
    "      >> r = numpy.uint8(32)\n"
    "\n"
    )
    .add_prototype("input, output", "array")
    .add_prototype("y", "r, g, b")
    .add_parameter("input", "array_like (uint8|uint16|float64, 2D)", "Input array containing an image with the shape ``(height, width)``")
    .add_parameter("output", "array (uint8|uint16|float64, 3D), optional", "Output array - if provided, should have matching data type to ``input``. The shape should be ``(3, height, width)``")
    .add_parameter("y", "scalar (uint8|uint16|float64)", "The gray-scale pixel scalar you wish to convert into an RGB tuple")
    .add_return("array", "array (uint8|uint16|float64, 3D)", "The ``output`` array is returned by the function. If one was not provided, a new one is allocated internally")
    .add_return("r, g, b", "scalar (uint8|uint16|float64)", "Discrete pixel values for the red, green and blue channels")
;

static PyMethodDef module_methods[] = {
    {
      s_rgb_to_gray.name(),
      (PyCFunction)PyBobIpColor_RgbToGray,
      METH_VARARGS|METH_KEYWORDS,
      s_rgb_to_gray.doc()
    },
    {
      s_gray_to_rgb.name(),
      (PyCFunction)PyBobIpColor_GrayToRgb,
      METH_VARARGS|METH_KEYWORDS,
      s_gray_to_rgb.doc()
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "Bob Image Processing Color Conversion");

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  XBOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
# else
  PyObject* m = Py_InitModule3(XBOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!m) return 0;
  auto m_ = make_safe(m); ///< protects against early returns

  if (PyModule_AddStringConstant(m, "__version__", XBOB_EXT_MODULE_VERSION) < 0)
    return 0;

  /* imports xbob.blitz C-API + dependencies */
  if (import_xbob_blitz() < 0) return 0;

  Py_INCREF(m);
  return m;

}

PyMODINIT_FUNC XBOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
