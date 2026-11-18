#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace astroviper {
template <typename T> inline bool naturally_aligned(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) == 0;
}

// Non-owning access to external contiguous storage. Unaligned loads/stores
// use a single scalar memcpy, never an array temporary. The aligned template
// specialization retains typed loads/stores and is selected once per call.
template <typename T, bool Aligned> class ScalarBuffer {
    using Value = std::remove_const_t<T>;
    using Byte = std::conditional_t<std::is_const_v<T>, const unsigned char, unsigned char>;
    Byte* data_;
public:
    using Pointer = std::conditional_t<std::is_const_v<T>, const void*, void*>;
    explicit ScalarBuffer(Pointer data) : data_(static_cast<Byte*>(data)) {}
    Value load(std::ptrdiff_t index) const {
        if constexpr (Aligned) return reinterpret_cast<const Value*>(data_)[index];
        Value value;
        std::memcpy(&value, data_ + index * sizeof(Value), sizeof(Value));
        return value;
    }
    void store(std::ptrdiff_t index, Value value) const {
        static_assert(!std::is_const_v<T>);
        if constexpr (Aligned) reinterpret_cast<Value*>(data_)[index] = value;
        else std::memcpy(data_ + index * sizeof(Value), &value, sizeof(Value));
    }
    class Reference {
        const ScalarBuffer& buffer_;
        std::ptrdiff_t index_;
    public:
        Reference(const ScalarBuffer& buffer, std::ptrdiff_t index) : buffer_(buffer), index_(index) {}
        operator Value() const { return buffer_.load(index_); }
        Reference& operator=(Value value) { buffer_.store(index_, value); return *this; }
        Reference& operator+=(Value value) { return *this = buffer_.load(index_) + value; }
    };
    auto operator[](std::ptrdiff_t index) const {
        if constexpr (std::is_const_v<T>) return load(index);
        else return Reference(*this, index);
    }
};
}  // namespace astroviper
