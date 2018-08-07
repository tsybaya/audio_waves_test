#include <windows.h>
#include <stdlib.h>

#include "wav.h"

enum FilterMode{FIR,IIR};

const char* filename_output = "output.wav";

int16_t* data = NULL, *pCoef_b = NULL, *pCoef_a = NULL, *pCoef_ba = NULL;

void deinit(void)
{
	free(data);
	free(pCoef_b);
	free(pCoef_a);
	free(pCoef_ba);
}

int main(int argc, char** argv)
{
	struct riff_header header;
	int num_coef_b=0, num_coef_a =0, num_coef_ba = 0;
	int16_t coef_a0 = 0;
	char* fname_coef_b, *fname_coef_a, *fname_wav;
	enum FilterMode filterMode = FIR;

	atexit(deinit);

	if (argc < 3) {
		fprintf(stderr, "Error! Not enought input arguments\n");
		fprintf(stderr, "Pass arguments in a next order [file.wav b_coeffs.txt a_coeffs.txt(optional)]");
		return -1;
	}

	if (argc > 4) {
		fprintf(stderr, "Error! Too many input arguments");
		fprintf(stderr, "Pass arguments in a next order: file.wav, b_coeffs.txt, a_coeffs.txt(optional)");
		return -1;
	}


	fname_wav = argv[1];
	fname_coef_b = argv[2];
	if (argc == 4) {
		fname_coef_a = argv[3];
		filterMode = IIR;
	}

	// read coefficients b from file
	pCoef_b = readCoefsFromFile(fname_coef_b, &num_coef_b);
	if (pCoef_b == NULL) {
		fprintf(stderr, "Error! Can not read b coeffs from file %s", fname_coef_b);
		return -1;
	}

	// read coefficients a from file
	if (filterMode == IIR) {
		pCoef_a = readCoefsFromFile(fname_coef_a, &num_coef_a);
		if (pCoef_a == NULL) {
			fprintf(stderr, "Error! Can not read a coeffs from file %s", fname_coef_a);
			return -1;
		}
	}

	// Write "b" and "a" coefficients in one array
	// in such_order: b1,b2,...,bn,a1,a2,...,am
	pCoef_ba=setCoefsBA(pCoef_b, num_coef_b, pCoef_a, num_coef_a, &num_coef_ba, &coef_a0);
	if (pCoef_ba == NULL) {
		fprintf(stderr, "Error! Can not set b and a coeffs");
		return -1;
	}

	// Read header
	if (readHeader(fname_wav, &header)) {
		fprintf(stderr,"Error! There are no riff header in file");
		return -1;
	}

	// Allocate memory for wav data
	data = (int16_t*)malloc(header.data_size);
	if (data == NULL) {
		fprintf(stderr, "Error! Can not allocate memory for data");
		return -1;
	}

	// Read data from file
	if (readData(fname_wav, &header, data)) {
		fprintf(stderr, "Error! Can not allocate memory for data");
		return -1;
	}

	// Filter signal
	if (filterMode == IIR) {
		if (filterDataIir(data, data, &header, pCoef_ba, num_coef_ba, num_coef_b, coef_a0)) {
			fprintf(stderr, "Error while filtering data");
			return -1;
		}
	}
	else if (filterMode == FIR)
		if (filterDataFir(data, data, &header, pCoef_ba, num_coef_ba)) {
			fprintf(stderr, "Error while filtering data");
			return -1;
		}

	// Write filtered data to wav
	if (writeWav(filename_output, &header, data)) {
		fprintf(stderr, "Error! Can not write wav file");
		return -1;
	}

	return 0;
}