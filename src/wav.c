#include "wav.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <mmintrin.h>

static int isHeaderRiff(struct riff_header* header);

int readHeader(const char* filename, struct riff_header* header)
{
	FILE* fp = NULL;

	const size_t elementCount = 1;

	if (filename == NULL || header == NULL)
		return -1;

	fp = fopen(filename, "rb");
	if (fp == NULL)
		return -1;

	fread(header->chunk_id, sizeof(header->chunk_id), elementCount, fp);
	fread(&header->chunk_size, sizeof(header->chunk_size), elementCount, fp);
	fread(header->format, sizeof(header->format), elementCount, fp);

	fread(header->subchunk1_id, sizeof(header->subchunk1_id), elementCount, fp);
	fread(&header->subchunk1_size, sizeof(header->subchunk1_size), elementCount, fp);
	fread(&header->audio_format, sizeof(header->audio_format), elementCount, fp);
	fread(&header->num_channels, sizeof(header->num_channels), elementCount, fp);
	fread(&header->sample_rate, sizeof(header->sample_rate), elementCount, fp);
	fread(&header->byte_rate, sizeof(header->byte_rate), elementCount, fp);
	fread(&header->block_align, sizeof(header->block_align), elementCount, fp);
	fread(&header->bits_per_sample, sizeof(header->bits_per_sample), elementCount, fp);

	fread(header->data_id, sizeof(header->data_id), elementCount, fp);
	fread(&header->data_size, sizeof(header->data_size), elementCount, fp);

	if (ferror(fp)) {
		fclose(fp);
		return -1;
	}

	fclose(fp);

	return isHeaderRiff(header);
}

int readData(const char* filename,const struct riff_header* header, int16_t* data)
{
	const long int OFFSET_RIFF_DATA = 44;

	FILE* fp;
	size_t res;
	size_t size = header->data_size;
	char *byteData = (char*)data;

	if (filename == NULL || header == NULL || data == NULL)
		return -1;

	fp = fopen(filename, "rb");
	if (fp == NULL)
		return -1;

	if (fseek(fp, OFFSET_RIFF_DATA, SEEK_SET)) {
		fclose(fp);
		return -1;
	}

	while ((res = fread(byteData, 1, size, fp)) > 0) {
		byteData += res;
		size -= res;
	}

	fclose(fp);

	return 0;
}

int filterDataIir(const int16_t* pData, int16_t* pOutput, const struct riff_header* pHeader,
	const int16_t* pCoef_ba, int num_coef_ba, int num_coef_b, int16_t coef_a0)
{	
	const int BIT_SHIFTING = 15;
	const int SHIFT_UPPER_BITS = 32;

	const int NUM_CHANNELS = pHeader->num_channels;
	const int DATA_LENGTH = pHeader->data_size / (pHeader->bits_per_sample / 8);

	const int NUM_LENGTH_SIMD = num_coef_ba / 4;
	const size_t SIZE_XY = num_coef_ba * sizeof(int16_t);
	const size_t SIZE_OFFSET_XY = (num_coef_ba - 1) * sizeof(int16_t);

	int i, j, k;
	int32_t sum;
	int16_t* p_xy=NULL, *p_xy_move=NULL, *p_y_prev=NULL;
	__m64 *_coef, *_xy, _res;

	if (pData == NULL    || pOutput == NULL  || pHeader == NULL ||
		pCoef_ba == NULL || num_coef_ba <= 0 || num_coef_b <= 0 ||
		coef_a0 == 0)
		return -1;

	p_xy = (int16_t*)malloc(SIZE_XY);
	p_xy_move = p_xy + 1;
	p_y_prev = p_xy + num_coef_b;

	_coef = (__m64*)pCoef_ba;
	_xy = (__m64*)p_xy;

	for (i = 0; i < NUM_CHANNELS; ++i)
	{
		memset(p_xy, 0, SIZE_XY);

		for (j = i; j < DATA_LENGTH; j += NUM_CHANNELS) 
		{
			p_xy[0] = pData[j];

			_res = _mm_setzero_si64();
			
			for (k = 0; k < NUM_LENGTH_SIMD; k++) {
				// _coef -> 4 * 16 bits coefs : c0,c1,c2,c3
				// _xy -> 4 * 16 bits previous values x and y: x0,x1,x2,x3
				//
				//
				// 1. _mm_madd_pi16(_mm_mulhi_pi16(_coef[k], _xy[k]), _ones)
				// multiply 16 bits each other, and unpacked it in two 32 bits
				// c0*x0 + c1*x1 | c2*x2 + c3*x3
				//
				// 2. _mm_srai_pi32(_mm_madd_pi16(_coef[k], _xy[k]), BIT_SHIFTING);
				// shift right "BIT_SHIFTING" bits
				//
				// 3. _mm_add_pi32(_mm_srai_pi32(_mm_madd_pi16(_coef[k], _xy[k]), BIT_SHIFTING), _res);
				// perform summation with previous res;

				_res = _mm_add_pi32(_mm_srai_pi32(_mm_madd_pi16(_coef[k], _xy[k]), BIT_SHIFTING), _res);
			}
	
			// 
			sum = _m_to_int(_res) + _m_to_int(_mm_srli_si64(_res, SHIFT_UPPER_BITS));

			pOutput[j] = (sum << BIT_SHIFTING) / coef_a0;

			memmove(p_xy_move, p_xy, SIZE_OFFSET_XY);
			*p_y_prev = pOutput[j];
		}
	}

	free(p_xy);
	_m_empty();
	return 0;
}

int filterDataFir(const int16_t* pData, int16_t* pOutput, const struct riff_header* pHeader,
	const int16_t* pCoef_b, int num_coef_b)
{
	const int BIT_SHIFTING = 15;
	const int SHIFT_UPPER_BITS = 32;

	const int NUM_CHANNELS = pHeader->num_channels;
	const int DATA_LENGTH = pHeader->data_size / (pHeader->bits_per_sample / 8);

	const int NUM_LENGTH_SIMD = num_coef_b / 4;
	const size_t SIZE_XY = num_coef_b * sizeof(int16_t);
	const size_t SIZE_OFFSET_XY = (num_coef_b - 1) * sizeof(int16_t);

	int i, j, k;
	int16_t* p_xy = NULL, *p_xy_move = NULL;
	__m64 *_coef, *_xy, _res;

	if (pData == NULL || pOutput == NULL || pHeader == NULL ||
		pCoef_b == NULL || num_coef_b <= 0)
		return -1;

	p_xy = (int16_t*)malloc(SIZE_XY);
	p_xy_move = p_xy + 1;

	_coef = (__m64*)pCoef_b;
	_xy = (__m64*)p_xy;

	for (i = 0; i < NUM_CHANNELS; ++i)
	{
		memset(p_xy, 0, SIZE_XY);

		for (j = i; j < DATA_LENGTH; j += NUM_CHANNELS)
		{
			p_xy[0] = pData[j];

			_res = _mm_setzero_si64();

			for (k = 0; k < NUM_LENGTH_SIMD; k++) {
				// _coef -> 4 * 16 bits coefs : c0,c1,c2,c3
				// _xy -> 4 * 16 bits previous values x and y: x0,x1,x2,x3
				//
				//
				// 1. _mm_madd_pi16(_mm_mulhi_pi16(_coef[k], _xy[k]), _ones)
				// multiply 16 bits each other, and unpacked it in two 32 bits
				// c0*x0 + c1*x1 | c2*x2 + c3*x3
				//
				// 2. _mm_srai_pi32(_mm_madd_pi16(_coef[k], _xy[k]), BIT_SHIFTING);
				// shift right "BIT_SHIFTING" bits
				//
				// 3. _mm_add_pi32(_mm_srai_pi32(_mm_madd_pi16(_coef[k], _xy[k]), BIT_SHIFTING), _res);
				// perform summation with previous res;

				_res = _mm_add_pi32(_mm_srai_pi32(_mm_madd_pi16(_coef[k], _xy[k]), BIT_SHIFTING), _res);
			}

			pOutput[j] = _m_to_int(_res) + _m_to_int(_mm_srli_si64(_res, SHIFT_UPPER_BITS));

			memmove(p_xy_move, p_xy, SIZE_OFFSET_XY);
		}
	}

	free(p_xy);
	_m_empty();
	return 0;
}

int writeWav(const char* filename,const struct riff_header* header,const int16_t* pData)
{
	FILE* fp;
	const size_t elementCount = 1;

	if (filename == NULL || header == NULL || pData == NULL)
		return -1;

	fp = fopen(filename, "wb");
	if (fp == NULL)
		return -1;

	fwrite(header->chunk_id, sizeof(header->chunk_id), elementCount, fp);
	fwrite(&header->chunk_size, sizeof(header->chunk_size), elementCount, fp);
	fwrite(header->format, sizeof(header->format), elementCount, fp);

	fwrite(header->subchunk1_id, sizeof(header->subchunk1_id), elementCount, fp);
	fwrite(&header->subchunk1_size, sizeof(header->subchunk1_size), elementCount, fp);
	fwrite(&header->audio_format, sizeof(header->audio_format), elementCount, fp);
	fwrite(&header->num_channels, sizeof(header->num_channels), elementCount, fp);
	fwrite(&header->sample_rate, sizeof(header->sample_rate), elementCount, fp);
	fwrite(&header->byte_rate, sizeof(header->byte_rate), elementCount, fp);
	fwrite(&header->block_align, sizeof(header->block_align), elementCount, fp);
	fwrite(&header->bits_per_sample, sizeof(header->bits_per_sample), elementCount, fp);

	fwrite(header->data_id, sizeof(header->data_id), elementCount, fp);
	fwrite(&header->data_size, sizeof(header->data_size), elementCount, fp);

	fwrite(pData, header->data_size, elementCount, fp);

	if (ferror(fp)) {
		fclose(fp);
		return -1;
	}
		
	fclose(fp);

	return 0;
}


int isHeaderRiff(struct riff_header* header)
{
	int res = -1;

	if (header == NULL)
		return res;

	if (header->chunk_id[0] == 'R' &&
		header->chunk_id[1] == 'I' &&
		header->chunk_id[2] == 'F' &&
		header->chunk_id[3] == 'F')
		res = 0;

	return res;
}

int16_t* setCoefsBA(const int16_t* pCoef_b, int num_coef_b,
	const int16_t* pCoef_a, int num_coef_a,
	int* pNum_coef_ba, int16_t* pCoef_a0)
{
	// Write "b" and "a" coefficients in array
	// in such_order: b0,b1,...,bn,a1,a2,...,am

	int16_t* pCoef_ba=NULL;
	int i;

	if (pCoef_b      == NULL || num_coef_b <= 0 ||
		pNum_coef_ba == NULL || pCoef_a0 == NULL)
		return NULL;

	if (pCoef_a == NULL && num_coef_a > 0)
		return NULL;

	if (pCoef_a != NULL && num_coef_a <= 0)
		return NULL;

	// Allocate memory for number of coefficients "b" and "a"
	// is a multiple of four plus 1 (4*n+1, n=1,2,3,..)
	// for SIMD 64 bit instructions.
	// Pad zeros for it.

	*pNum_coef_ba = num_coef_a+num_coef_b-1; //without coef a0
	if (*pNum_coef_ba % 4 != 0 || *pNum_coef_ba==0)
		*pNum_coef_ba += 4 - *pNum_coef_ba % 4;

	pCoef_ba = (int16_t*)malloc(*pNum_coef_ba * sizeof(int16_t));
	memset(pCoef_ba, 0, *pNum_coef_ba * sizeof(int16_t));

	if (num_coef_a!=0)
		*pCoef_a0 = pCoef_a[0];

	for (i = 0; i < num_coef_b; ++i)
		pCoef_ba[i] = pCoef_b[i];

	// Change sign for a coeffients to substitute an substract with an summation
	for (i = 1; i < num_coef_a; ++i) 
		pCoef_ba[num_coef_b + i - 1] = pCoef_a[i]==INT16_MIN ? -(pCoef_a[i]+1) : -pCoef_a[i];

	return pCoef_ba;
}

int16_t* readCoefsFromFile(const char* filename, int* num_coef)
{
	FILE* fp = NULL;
	int coef = 0;
	int16_t* pCoeff;
	int i = 0;
	
	if (filename == NULL)
		return NULL;

	*num_coef = 0;

	fp=fopen(filename, "r");
	if (fp == NULL)
		return NULL;

	// Get the number of coefficients
	while (!feof(fp)) {
		if (fscanf(fp, "%d", &coef) == 0)
			return NULL;
		++*num_coef;
	}
	fclose(fp);
	fp = NULL;

	fp = fopen(filename, "r");
	if (fp == NULL)
		return NULL;

	pCoeff = (int16_t*)malloc(*num_coef * sizeof(int16_t));
	while (!feof(fp)) {
		fscanf(fp, "%d", &coef);
		pCoeff[i++] = (int16_t)coef;
	}
	fclose(fp);
	fp = NULL;

	return pCoeff;
}