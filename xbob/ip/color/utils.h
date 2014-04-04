/**
 * @author André Anjos <andre.anjos@idiap.ch>
 * @date Fri  4 Apr 15:20:24 2014 CEST
 *
 * @brief Helpers for color conversion
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */


#include <xbob.blitz/cppapi.h>
#include <xbob.blitz/cleanup.h>

int check_and_allocate(Py_ssize_t input_dims, Py_ssize_t output_dims,
    boost::shared_ptr<PyBlitzArrayObject>& input,
    boost::shared_ptr<PyBlitzArrayObject>& output);
