#ifndef UTILS_H
#define UTILS_H

#include <stdexcept>
#include <iostream>


#define ASSERT(condition, msg) \
    if (!(condition)) { throw DetailedException(msg, __PRETTY_FUNCTION__); }


class DetailedException : public std::exception {
    std::string message;
    const char* func_name;
    public:
        DetailedException(const char* msg, const char* func) : message(msg), func_name(func) {}
        const char* what() const noexcept override { return message.c_str(); }
        const char* function() const noexcept { return func_name; }
};


#endif // UTILS_H