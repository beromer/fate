#ifndef BLOCK_H
#define BLOCK_H

#include "rkfunction.hpp"
#include <string>
#include "kokkosTypes.hpp"
#include "hdf5.h"

template <typename T>
class blockWriter {
  public:
    blockWriter();
    blockWriter(struct inputConfig&,std::unique_ptr<class rk_func>&,std::string,std::string,bool,size_t,bool);
    blockWriter(struct inputConfig&,std::unique_ptr<class rk_func>&,std::string,std::string,bool,size_t,
                                  std::vector<size_t>,std::vector<size_t>,std::vector<size_t>,bool);
    void write(struct inputConfig cf, std::unique_ptr<class rk_func>&f, int tdx, FSCAL time);
    size_t frq();

  private:
      
    std::vector<size_t> lStart;  // local starting index
    std::vector<size_t> lEnd;    // local ending cell index
    std::vector<size_t> lEndG;   // local ending grid index
    std::vector<size_t> lExt;    // local extent
    std::vector<size_t> lExtG;   // local grid extent
    std::vector<size_t> lOffset; // local offset
    std::vector<FSCAL> gOrigin; // global origin of block
    std::vector<size_t> gStart;  // global starting index
    std::vector<size_t> gEnd;    // global ending index
    std::vector<size_t> gExt;    // global extent
    std::vector<size_t> gExtG;   // global grid extent
    std::vector<size_t> stride;  // slice stride
    std::vector<FSCAL> iodx;

    bool chunkable;

    size_t freq;    // block write frequency

    bool writeVarx;
    
    size_t lElems;
    size_t lElemsG;

    int myColor;

    std::vector<T> varData;
    std::vector<T> gridData;

    FS4DH gridH;
    FS4DH varH;
    FS4DH varxH;

    bool slicePresent;
    int offsetDelta;

    std::string path;
    std::string name;
    int pad;
    bool avg;

    bool appStep;

    void write_h5(hid_t, std::string, int, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>, std::vector<T>&);

    void dataPack(int, int, const std::vector<size_t>&, const std::vector<size_t>&, const std::vector<size_t>&,
                            const std::vector<size_t>&, std::vector<T>&, const FS4DH&, const int, const bool);
    
    hid_t openHDF5ForWrite(std::string);
    void close_h5(hid_t);
};

template <typename H>
hid_t getH5Type();

#endif
