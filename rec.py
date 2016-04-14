import theano, numpy
import theano.tensor as T


arr = T.ivector('arr')
#x1 = T.scalar('x1')
#x2 = T.scalar('x2')

def sum_prod(x, rec):
    #new_rec = T.inc_subtensor(rec[0], x + rec[0])
    #new_rec = T.set_subtensor(new_rec[1], x * new_rec[1])
    rec[0] = rec[0] + x
    return rec

result, proc = theano.scan(fn=sum_prod,
        outputs_info=T.as_tensor_variable(numpy.asarray([0, 1], dtype=numpy.int32)),
        sequences=arr)

sp = theano.function(inputs=[arr], outputs=result)
print(sp(numpy.asarray([1,2,3,4],dtype=numpy.int32)))

