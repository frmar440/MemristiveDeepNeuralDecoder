
#include "weight_probe.h"
#include "math_util.h"
#include "utility_functions.h"
#include <limits>

namespace RPU {

/***********************************************************/
// ctors

template <typename T>
WeightProbe<T>::WeightProbe(int x_size, int d_size)
    : x_size_(x_size), d_size_(d_size), size_(d_size * x_size) {}

template <typename T> void WeightProbe<T>::probe(T *weights, T *probe_weights) {

  PRAGMA_SIMD
  for (int i = 0; i < size_; ++i) {
    weights[i] *= probe_weights[i];
  }
}

template <typename T> void WeightProbe<T>::apply(T *weights, T *probe_weights) {

  // do the probing
  probe(weights, probe_weights);
}

template class WeightProbe<float>;
#ifdef RPU_USE_DOUBLE
template class WeightProbe<double>;
#endif

} // namespace RPU
