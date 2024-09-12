#pragma once

#include <sstream>

namespace intel_npu {

class CustomStringBuf : public std::stringbuf {
public:
    CustomStringBuf(std::string&& str) {
        this->_str = std::move(str);
        this->setp(&this->_str[0], &this->_str[0] + this->_str.size());
        this->setg(&this->_str[0], &this->_str[1], &this->_str[0] + this->_str.size());
        this->pbump(this->_str.size());
    }
    
    std::string&& str() {
        return std::move(_str);
    }
private:
    std::string _str;
};

}  // namespace intel_npu

namespace std {

template<>
ostringstream::_Mystr ostringstream::str() const {
    intel_npu::CustomStringBuf* customStringBufPtr = dynamic_cast<intel_npu::CustomStringBuf*>(ostream::rdbuf());
    if (customStringBufPtr != nullptr) {
       return customStringBufPtr->str();
    } else {
       return this->rdbuf()->str();
    }
}

}  // namespace std