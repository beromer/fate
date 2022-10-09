#include <string>
#include <vector>
#include "hdf5.h"

template<typename T>
class h5Writer {

  public:
    h5Writer();
    void open(std::string fname);
    void openRead(std::string fname);

    void close();

    void write(std::string, int, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>, std::vector<T>&, bool, bool);

    template<typename S>
    void writeAttribute(std::string, S data);

    void read(std::string path, int ndim, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>, T*);

    template<typename S>
    void readAttribute(std::string, S& data);

    void openGroup(std::string name);
    void closeGroup();

  private:
    void checkDataDimensions(hid_t, int ndim, std::vector<hsize_t>);
    std::string filename;
    hid_t file_id;
    hid_t group_id;
};
