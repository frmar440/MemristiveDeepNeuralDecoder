
#pragma once

#include "rng.h"
#include <memory>

namespace RPU {

template <typename T> class WeightProbe {

public:
  explicit WeightProbe(int x_size, int d_size);
  WeightProbe(){};

  /* in-place probing of weights */
  void apply(T *weights, T *probe_weights);

private:
  void probe(T *weights, T *probe_weights);

  int x_size_ = 0;
  int d_size_ = 0;
  int size_ = 0;
};

} // namespace RPU
