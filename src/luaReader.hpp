#ifndef LUAREADER_H
#define LUAREADER_H

#include <string>
#include "lua.hpp"
#include <vector>
#include "block.hpp"
#include "input.hpp"
#include "rkfunction.hpp"

class luaReader {

public:
  luaReader(std::string fname,std::string root);
  void close();

  template <class T>
  void getArray(std::string,T*,int);
 
  template <class T>
  void getArray(std::vector<T>&,int);

  void getSpeciesData(struct inputConfig&);
  void getIOBlock(struct inputConfig&, std::unique_ptr<class rk_func>&, int, std::vector<blockWriter<float>>&);

  template <class T>
  void get(std::initializer_list<std::string> keys, T& n);

  template <class T, class S>
  void get(std::initializer_list<std::string> keys, T& n, S d);

  template <class T>
  void get(std::initializer_list<std::string> keys, std::vector<T>& v, int n);

  template <class T>
  void getValue(T& n);

  FSCAL call(std::string, int ,...);

private:
  lua_State *L;

  std::string root;
  std::string filename;

  void error(lua_State *, const char *, ...);

  bool undefined(std::string key);

  bool getBool(std::string);
  int getInt(std::string);
  FSCAL getDouble(std::string);
  std::string getString(std::string);

};

#endif
