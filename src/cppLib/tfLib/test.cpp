#include <cstdint>

#include "absl/status/status.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/public/version.h"

using namespace tensorflow;

REGISTER_OP("TestOp")
    .Input("in: double")
    .Output("out: double")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
#if TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION < 10
  return Status::OK();
#else
  return absl::OkStatus();
#endif
    });

#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class TestOp : public OpKernel {
 public:
  explicit TestOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<double>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<double>();

    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = input(i)*2.0;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("TestOp").Device(DEVICE_CPU), TestOp);

