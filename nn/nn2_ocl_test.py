from nn2_ocl import *

class testCL(CL):
    def test(self):
        
        for i in xrange(10):
            S = np.int32(100)
            N = np.int32(100)
            M = np.int32(100)
            outsize = M*S*4

            a1 = np.matrix(np.random.rand(S,N), np.float32)
            w = np.matrix(np.random.rand(N,M), np.float32)

            z2 = np.matrix(np.zeros((S,M)), np.float32)
            a2 = np.matrix(np.zeros((S,M)), np.float32)


            z2_cpu = a1 * w
            a2_cpu = sigmoid_matrix(z2_cpu)
            mf = cl.mem_flags

            d_a1 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = a1)
            d_z2 = cl.Buffer(self.ctx, mf.WRITE_ONLY, size = outsize)
            d_a2 = cl.Buffer(self.ctx, mf.WRITE_ONLY, size = outsize)
            d_w = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = w)


            e = self.prg.forward_kernel(self.queue, (S,M), None, d_a1, d_z2, d_a2, d_w, N, M, S)
            e.wait()
            cl.enqueue_copy(self.queue, a2, d_a2).wait()
            cl.enqueue_copy(self.queue, z2, d_z2).wait()
            

            adiff = (a2-a2_cpu)
            zdiff = (z2-z2_cpu)
            zsq = np.sum(np.multiply(zdiff, zdiff))
            asq = np.sum(np.multiply(adiff, adiff))
            print("adiff: {} zdiff: {}".format(asq, zsq))

if __name__=="__main__":
    testCL().test()

