#define rmo(X,Y,W) (X*W+Y)

float sigmoid(float z) {
	z = clamp(z, -88.0F, 700.0F);
	return 1.0F / (1.0F + exp(-z));
}

float sigmoid_grad(float z) {
	float s = sigmoid(z);
	return s*(1-s);
}

float matrix_mul(__global float *a, __global float *b, int N, int M, int S) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	float v = 0.0F;
	for (int k = 0; k< N; k++ ) {
		v += a[rmo(i,k,N)] * b[rmo(k,j,M)];
	}
	return v;
}

float matrix_mul_2t(__global float *a, __global float *b, int N, int M, int S) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	float v = 0.0F;
	for (int k = 0; k<M; k++ ) {
		v += a[rmo(i,k,M)] * b[rmo(j,k,M)];
	}
	return v;
}

float matrix_mul_1t(__global float *a, __global float *b, int N, int M, int S) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	float v = 0.0F;
	for (int k = 0; k< S; k++ ) {
		v += a[rmo(k,i,N)] * b[rmo(k,j,M)];
	}
	return v;
}

__kernel void forward_kernel(__global float *a1, 
				    __global float *z2, 
				    __global float *a2, 
				    __global float *w,
				    int N, int M, int S) {
	//a1 sxN
	//w NxM
	//z2 sxM
	//a2 sxM

	//z2 = a1 * w
	//a2 = sigmoid(z2)

	int i = get_global_id(0);
	int j = get_global_id(1);
	
	float v = matrix_mul(a1, w, N, M, S);
	z2[rmo(i,j,M)] = v;
	a2[rmo(i,j,M)] = sigmoid(v);
}

__kernel void backward_kernel(__global float *d1, __global float *d2, 
				       __global float *w, 
				       __global float *z1,
				       int N, int M, int S) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	float v = matrix_mul_2t(d2, w, N, M, S);

	d1[rmo(i,j,N)] = v * sigmoid_grad(z1[rmo(i, j, N)]);
	
}

__kernel void acc_grad(__global float *d2, __global float *a1, __global float *g, int N, int M, int S) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	g[rmo(i,j,M)] = matrix_mul_1t(a1, d2, N, M, S) / S;
}

__kernel void compute_last_deltas(__global float *fv, __global float *tv, __global float *deltas, int M, int S) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	deltas[rmo(i,j,M)] = fv[rmo(i,j,M)] - tv[rmo(i,j,M)];
}

__kernel void lr_cost(__global float *a, __global float *tv, __global float *out, int M, int S) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	
	float y = tv[rmo(i,j,M)];
	float v = a[rmo(i,j,M)];
	v = clamp(v, 0.00001F, 1.0F - 0.00001F);

	out[rmo(i,j,M)] =  -y*log(v) - (1-y)*log(1-v);

}
