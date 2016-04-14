import theano.tensor as T
import theano, numpy

"""
Order of scan fn arguments
1. sequences (if any)
2. prior results(s) if needed
3. non-sequences
"""

"""
Example 1 Simple loop with accumulation
Computing A^k
"""
k = T.iscalar('k')
A = T.vector('A')
result, updates = theano.scan(fn=lambda prior, b: prior * b,
        outputs_info=T.ones_like(A),
        non_sequences=A,
        n_steps=k)

power = theano.function(inputs=[A, k], outputs = result[-1], updates=updates)
# print(power([1,2,3], 3))

"""
Example 2 Iterating over the first dimension of a tensor
Calculating a polynomial
"""
coefficients = T.vector('cofficients')
x = T.scalar('x')
max_coefficients_supported = 10000

components, updates = theano.scan(
        fn=lambda coefficient, power, free_var: coefficient * (free_var ** power),
        outputs_info=None,
        sequences=[coefficients, T.arange(max_coefficients_supported)],
        non_sequences=x)

calculate_polynomial = theano.function(inputs=[coefficients, x],
        outputs=T.sum(components))

# print(calculate_polynomial(numpy.ones(10, dtype=theano.config.floatX), 2))

"""
Example 3 Simple accumulation into a scalar, ditching lambda
"""
up_to = T.iscalar('up_to')

def accumulate_by_adding(val, result):
    return result + val
seq = T.arange(up_to)
outputs_info = T.as_tensor_variable(numpy.asarray(0, seq.dtype))
result, updates = theano.scan(fn=accumulate_by_adding,
        outputs_info=outputs_info,
        sequences=seq)

triangular_seq = theano.function(inputs=[up_to], outputs=result)
# print(triangular_seq(12))


"""
Example 4 Matrix setting by index
"""
location = T.imatrix("location")
values = T.fvector("values")
output_model = T.fmatrix("output_model")

def set_val_at_pos(location, val, out):
    zeros = T.zeros_like(out)
    zeros_subtensor = zeros[location[0], location[1]]
    return T.set_subtensor(zeros_subtensor, val)

result, updates = theano.scan(fn=set_val_at_pos,
        outputs_info=None,
        sequences=[location, values],
        non_sequences=output_model)

assign_values_at_position = theano.function(inputs=[location, values, output_model],
        outputs=result[-1],
        updates=updates)

"""
print(assign_values_at_position(
    numpy.asarray([[0,0],[1,1],[2,2]], dtype=numpy.int32),
    numpy.asarray([1,2,3], dtype=theano.config.floatX),
    numpy.zeros((5,5), dtype=theano.config.floatX)
    ))
"""

"""
Example 5
Scan and accumulate over a vector
"""
x = T.fvector('x')
result, updates = theano.scan(
        fn=lambda val, prior: val + prior,
        outputs_info=T.as_tensor_variable(numpy.asarray(0, dtype=theano.config.floatX)),
        sequences=x)

sum_func = theano.function(inputs=[x], outputs=result[-1])
print(sum_func([1,2,3,5,7,2,7,8,2,3,4]))
