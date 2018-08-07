#ifndef WAV_H
#define WAV_H

#include <stdio.h>
#include <stdint.h>

struct riff_header
{
	// Ther "RIFF" chunk descriptor
	char    chunk_id[4];          
	uint32_t chunk_size;    
	char    format[4];

	// The "fmt" sub-chunk
	char    subchunk1_id[4];         
	uint32_t subchunk1_size;           
	uint16_t audio_format;
	uint16_t num_channels;
	uint32_t sample_rate;      
	uint32_t byte_rate;
	uint16_t block_align;
	uint16_t bits_per_sample;

	// The "data" sub-chunk
	char    data_id[4];      
	uint32_t data_size;
};

int readHeader(const char* filename,struct riff_header* header);
int readData(const char* filename, const struct riff_header* header, int16_t* data);

int16_t* setCoefsBA(const int16_t* pCoef_b, int num_coef_b,
	const int16_t* pCoef_a, int num_coef_a,
	int* num_coef_ba, int16_t* pCoef_a0);

int filterDataIir(const int16_t* pData, int16_t* pOutput, const struct riff_header* pHeader,
	const int16_t* pCoef_ba, int num_coef_ba, int num_coef_b, int16_t coef_a0);

int filterDataFir(const int16_t* pData, int16_t* pOutput, const struct riff_header* pHeader,
	const int16_t* pCoef_b, int num_coef_b);

int writeWav(const char* filename,const struct riff_header* header,const int16_t* pData);

int16_t* readCoefsFromFile(const char* filename, int* num_coef);

#endif WAV_HEADER_H