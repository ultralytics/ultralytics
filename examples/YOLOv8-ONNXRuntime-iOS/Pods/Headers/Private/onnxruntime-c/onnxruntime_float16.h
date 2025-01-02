// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <cmath>
#include <cstring>
#include <limits>

namespace onnxruntime_float16 {

namespace detail {

enum class endian {
#if defined(_WIN32)
  little = 0,
  big = 1,
  native = little,
#elif defined(__GNUC__) || defined(__clang__)
  little = __ORDER_LITTLE_ENDIAN__,
  big = __ORDER_BIG_ENDIAN__,
  native = __BYTE_ORDER__,
#else
#error onnxruntime_float16::detail::endian is not implemented in this environment.
#endif
};

static_assert(
    endian::native == endian::little || endian::native == endian::big,
    "Only little-endian or big-endian native byte orders are supported.");

}  // namespace detail

/// <summary>
/// Shared implementation between public and internal classes. CRTP pattern.
/// </summary>
template <class Derived>
struct Float16Impl {
 protected:
  /// <summary>
  /// Converts from float to uint16_t float16 representation
  /// </summary>
  /// <param name="v"></param>
  /// <returns></returns>
  constexpr static uint16_t ToUint16Impl(float v) noexcept;

  /// <summary>
  /// Converts float16 to float
  /// </summary>
  /// <returns>float representation of float16 value</returns>
  float ToFloatImpl() const noexcept;

  /// <summary>
  /// Creates an instance that represents absolute value.
  /// </summary>
  /// <returns>Absolute value</returns>
  uint16_t AbsImpl() const noexcept {
    return static_cast<uint16_t>(val & ~kSignMask);
  }

  /// <summary>
  /// Creates a new instance with the sign flipped.
  /// </summary>
  /// <returns>Flipped sign instance</returns>
  uint16_t NegateImpl() const noexcept {
    return IsNaN() ? val : static_cast<uint16_t>(val ^ kSignMask);
  }

 public:
  // uint16_t special values
  static constexpr uint16_t kSignMask = 0x8000U;
  static constexpr uint16_t kBiasedExponentMask = 0x7C00U;
  static constexpr uint16_t kPositiveInfinityBits = 0x7C00U;
  static constexpr uint16_t kNegativeInfinityBits = 0xFC00U;
  static constexpr uint16_t kPositiveQNaNBits = 0x7E00U;
  static constexpr uint16_t kNegativeQNaNBits = 0xFE00U;
  static constexpr uint16_t kEpsilonBits = 0x4170U;
  static constexpr uint16_t kMinValueBits = 0xFBFFU;  // Minimum normal number
  static constexpr uint16_t kMaxValueBits = 0x7BFFU;  // Largest normal number
  static constexpr uint16_t kOneBits = 0x3C00U;
  static constexpr uint16_t kMinusOneBits = 0xBC00U;

  uint16_t val{0};

  Float16Impl() = default;

  /// <summary>
  /// Checks if the value is negative
  /// </summary>
  /// <returns>true if negative</returns>
  bool IsNegative() const noexcept {
    return static_cast<int16_t>(val) < 0;
  }

  /// <summary>
  /// Tests if the value is NaN
  /// </summary>
  /// <returns>true if NaN</returns>
  bool IsNaN() const noexcept {
    return AbsImpl() > kPositiveInfinityBits;
  }

  /// <summary>
  /// Tests if the value is finite
  /// </summary>
  /// <returns>true if finite</returns>
  bool IsFinite() const noexcept {
    return AbsImpl() < kPositiveInfinityBits;
  }

  /// <summary>
  /// Tests if the value represents positive infinity.
  /// </summary>
  /// <returns>true if positive infinity</returns>
  bool IsPositiveInfinity() const noexcept {
    return val == kPositiveInfinityBits;
  }

  /// <summary>
  /// Tests if the value represents negative infinity
  /// </summary>
  /// <returns>true if negative infinity</returns>
  bool IsNegativeInfinity() const noexcept {
    return val == kNegativeInfinityBits;
  }

  /// <summary>
  /// Tests if the value is either positive or negative infinity.
  /// </summary>
  /// <returns>True if absolute value is infinity</returns>
  bool IsInfinity() const noexcept {
    return AbsImpl() == kPositiveInfinityBits;
  }

  /// <summary>
  /// Tests if the value is NaN or zero. Useful for comparisons.
  /// </summary>
  /// <returns>True if NaN or zero.</returns>
  bool IsNaNOrZero() const noexcept {
    auto abs = AbsImpl();
    return (abs == 0 || abs > kPositiveInfinityBits);
  }

  /// <summary>
  /// Tests if the value is normal (not zero, subnormal, infinite, or NaN).
  /// </summary>
  /// <returns>True if so</returns>
  bool IsNormal() const noexcept {
    auto abs = AbsImpl();
    return (abs < kPositiveInfinityBits)           // is finite
           && (abs != 0)                           // is not zero
           && ((abs & kBiasedExponentMask) != 0);  // is not subnormal (has a non-zero exponent)
  }

  /// <summary>
  /// Tests if the value is subnormal (denormal).
  /// </summary>
  /// <returns>True if so</returns>
  bool IsSubnormal() const noexcept {
    auto abs = AbsImpl();
    return (abs < kPositiveInfinityBits)           // is finite
           && (abs != 0)                           // is not zero
           && ((abs & kBiasedExponentMask) == 0);  // is subnormal (has a zero exponent)
  }

  /// <summary>
  /// Creates an instance that represents absolute value.
  /// </summary>
  /// <returns>Absolute value</returns>
  Derived Abs() const noexcept { return Derived::FromBits(AbsImpl()); }

  /// <summary>
  /// Creates a new instance with the sign flipped.
  /// </summary>
  /// <returns>Flipped sign instance</returns>
  Derived Negate() const noexcept { return Derived::FromBits(NegateImpl()); }

  /// <summary>
  /// IEEE defines that positive and negative zero are equal, this gives us a quick equality check
  /// for two values by or'ing the private bits together and stripping the sign. They are both zero,
  /// and therefore equivalent, if the resulting value is still zero.
  /// </summary>
  /// <param name="lhs">first value</param>
  /// <param name="rhs">second value</param>
  /// <returns>True if both arguments represent zero</returns>
  static bool AreZero(const Float16Impl& lhs, const Float16Impl& rhs) noexcept {
    return static_cast<uint16_t>((lhs.val | rhs.val) & ~kSignMask) == 0;
  }

  bool operator==(const Float16Impl& rhs) const noexcept {
    if (IsNaN() || rhs.IsNaN()) {
      // IEEE defines that NaN is not equal to anything, including itself.
      return false;
    }
    return val == rhs.val;
  }

  bool operator!=(const Float16Impl& rhs) const noexcept { return !(*this == rhs); }

  bool operator<(const Float16Impl& rhs) const noexcept {
    if (IsNaN() || rhs.IsNaN()) {
      // IEEE defines that NaN is unordered with respect to everything, including itself.
      return false;
    }

    const bool left_is_negative = IsNegative();
    if (left_is_negative != rhs.IsNegative()) {
      // When the signs of left and right differ, we know that left is less than right if it is
      // the negative value. The exception to this is if both values are zero, in which case IEEE
      // says they should be equal, even if the signs differ.
      return left_is_negative && !AreZero(*this, rhs);
    }
    return (val != rhs.val) && ((val < rhs.val) ^ left_is_negative);
  }
};

// The following Float16_t conversions are based on the code from
// Eigen library.

// The conversion routines are Copyright (c) Fabian Giesen, 2016.
// The original license follows:
//
// Copyright (c) Fabian Giesen, 2016
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

namespace detail {
union float32_bits {
  unsigned int u;
  float f;
};
}  // namespace detail

template <class Derived>
inline constexpr uint16_t Float16Impl<Derived>::ToUint16Impl(float v) noexcept {
  detail::float32_bits f{};
  f.f = v;

  constexpr detail::float32_bits f32infty = {255 << 23};
  constexpr detail::float32_bits f16max = {(127 + 16) << 23};
  constexpr detail::float32_bits denorm_magic = {((127 - 15) + (23 - 10) + 1) << 23};
  constexpr unsigned int sign_mask = 0x80000000u;
  uint16_t val = static_cast<uint16_t>(0x0u);

  unsigned int sign = f.u & sign_mask;
  f.u ^= sign;

  // NOTE all the integer compares in this function can be safely
  // compiled into signed compares since all operands are below
  // 0x80000000. Important if you want fast straight SSE2 code
  // (since there's no unsigned PCMPGTD).

  if (f.u >= f16max.u) {                         // result is Inf or NaN (all exponent bits set)
    val = (f.u > f32infty.u) ? 0x7e00 : 0x7c00;  // NaN->qNaN and Inf->Inf
  } else {                                       // (De)normalized number or zero
    if (f.u < (113 << 23)) {                     // resulting FP16 is subnormal or zero
      // use a magic value to align our 10 mantissa bits at the bottom of
      // the float. as long as FP addition is round-to-nearest-even this
      // just works.
      f.f += denorm_magic.f;

      // and one integer subtract of the bias later, we have our final float!
      val = static_cast<uint16_t>(f.u - denorm_magic.u);
    } else {
      unsigned int mant_odd = (f.u >> 13) & 1;  // resulting mantissa is odd

      // update exponent, rounding bias part 1
      // Equivalent to `f.u += ((unsigned int)(15 - 127) << 23) + 0xfff`, but
      // without arithmetic overflow.
      f.u += 0xc8000fffU;
      // rounding bias part 2
      f.u += mant_odd;
      // take the bits!
      val = static_cast<uint16_t>(f.u >> 13);
    }
  }

  val |= static_cast<uint16_t>(sign >> 16);
  return val;
}

template <class Derived>
inline float Float16Impl<Derived>::ToFloatImpl() const noexcept {
  constexpr detail::float32_bits magic = {113 << 23};
  constexpr unsigned int shifted_exp = 0x7c00 << 13;  // exponent mask after shift
  detail::float32_bits o{};

  o.u = (val & 0x7fff) << 13;            // exponent/mantissa bits
  unsigned int exp = shifted_exp & o.u;  // just the exponent
  o.u += (127 - 15) << 23;               // exponent adjust

  // handle exponent special cases
  if (exp == shifted_exp) {   // Inf/NaN?
    o.u += (128 - 16) << 23;  // extra exp adjust
  } else if (exp == 0) {      // Zero/Denormal?
    o.u += 1 << 23;           // extra exp adjust
    o.f -= magic.f;           // re-normalize
  }

  // Attempt to workaround the Internal Compiler Error on ARM64
  // for bitwise | operator, including std::bitset
#if (defined _MSC_VER) && (defined _M_ARM || defined _M_ARM64 || defined _M_ARM64EC)
  if (IsNegative()) {
    return -o.f;
  }
#else
  // original code:
  o.u |= (val & 0x8000U) << 16U;  // sign bit
#endif
  return o.f;
}

/// Shared implementation between public and internal classes. CRTP pattern.
template <class Derived>
struct BFloat16Impl {
 protected:
  /// <summary>
  /// Converts from float to uint16_t float16 representation
  /// </summary>
  /// <param name="v"></param>
  /// <returns></returns>
  static uint16_t ToUint16Impl(float v) noexcept;

  /// <summary>
  /// Converts bfloat16 to float
  /// </summary>
  /// <returns>float representation of bfloat16 value</returns>
  float ToFloatImpl() const noexcept;

  /// <summary>
  /// Creates an instance that represents absolute value.
  /// </summary>
  /// <returns>Absolute value</returns>
  uint16_t AbsImpl() const noexcept {
    return static_cast<uint16_t>(val & ~kSignMask);
  }

  /// <summary>
  /// Creates a new instance with the sign flipped.
  /// </summary>
  /// <returns>Flipped sign instance</returns>
  uint16_t NegateImpl() const noexcept {
    return IsNaN() ? val : static_cast<uint16_t>(val ^ kSignMask);
  }

 public:
  // uint16_t special values
  static constexpr uint16_t kSignMask = 0x8000U;
  static constexpr uint16_t kBiasedExponentMask = 0x7F80U;
  static constexpr uint16_t kPositiveInfinityBits = 0x7F80U;
  static constexpr uint16_t kNegativeInfinityBits = 0xFF80U;
  static constexpr uint16_t kPositiveQNaNBits = 0x7FC1U;
  static constexpr uint16_t kNegativeQNaNBits = 0xFFC1U;
  static constexpr uint16_t kSignaling_NaNBits = 0x7F80U;
  static constexpr uint16_t kEpsilonBits = 0x0080U;
  static constexpr uint16_t kMinValueBits = 0xFF7FU;
  static constexpr uint16_t kMaxValueBits = 0x7F7FU;
  static constexpr uint16_t kRoundToNearest = 0x7FFFU;
  static constexpr uint16_t kOneBits = 0x3F80U;
  static constexpr uint16_t kMinusOneBits = 0xBF80U;

  uint16_t val{0};

  BFloat16Impl() = default;

  /// <summary>
  /// Checks if the value is negative
  /// </summary>
  /// <returns>true if negative</returns>
  bool IsNegative() const noexcept {
    return static_cast<int16_t>(val) < 0;
  }

  /// <summary>
  /// Tests if the value is NaN
  /// </summary>
  /// <returns>true if NaN</returns>
  bool IsNaN() const noexcept {
    return AbsImpl() > kPositiveInfinityBits;
  }

  /// <summary>
  /// Tests if the value is finite
  /// </summary>
  /// <returns>true if finite</returns>
  bool IsFinite() const noexcept {
    return AbsImpl() < kPositiveInfinityBits;
  }

  /// <summary>
  /// Tests if the value represents positive infinity.
  /// </summary>
  /// <returns>true if positive infinity</returns>
  bool IsPositiveInfinity() const noexcept {
    return val == kPositiveInfinityBits;
  }

  /// <summary>
  /// Tests if the value represents negative infinity
  /// </summary>
  /// <returns>true if negative infinity</returns>
  bool IsNegativeInfinity() const noexcept {
    return val == kNegativeInfinityBits;
  }

  /// <summary>
  /// Tests if the value is either positive or negative infinity.
  /// </summary>
  /// <returns>True if absolute value is infinity</returns>
  bool IsInfinity() const noexcept {
    return AbsImpl() == kPositiveInfinityBits;
  }

  /// <summary>
  /// Tests if the value is NaN or zero. Useful for comparisons.
  /// </summary>
  /// <returns>True if NaN or zero.</returns>
  bool IsNaNOrZero() const noexcept {
    auto abs = AbsImpl();
    return (abs == 0 || abs > kPositiveInfinityBits);
  }

  /// <summary>
  /// Tests if the value is normal (not zero, subnormal, infinite, or NaN).
  /// </summary>
  /// <returns>True if so</returns>
  bool IsNormal() const noexcept {
    auto abs = AbsImpl();
    return (abs < kPositiveInfinityBits)           // is finite
           && (abs != 0)                           // is not zero
           && ((abs & kBiasedExponentMask) != 0);  // is not subnormal (has a non-zero exponent)
  }

  /// <summary>
  /// Tests if the value is subnormal (denormal).
  /// </summary>
  /// <returns>True if so</returns>
  bool IsSubnormal() const noexcept {
    auto abs = AbsImpl();
    return (abs < kPositiveInfinityBits)           // is finite
           && (abs != 0)                           // is not zero
           && ((abs & kBiasedExponentMask) == 0);  // is subnormal (has a zero exponent)
  }

  /// <summary>
  /// Creates an instance that represents absolute value.
  /// </summary>
  /// <returns>Absolute value</returns>
  Derived Abs() const noexcept { return Derived::FromBits(AbsImpl()); }

  /// <summary>
  /// Creates a new instance with the sign flipped.
  /// </summary>
  /// <returns>Flipped sign instance</returns>
  Derived Negate() const noexcept { return Derived::FromBits(NegateImpl()); }

  /// <summary>
  /// IEEE defines that positive and negative zero are equal, this gives us a quick equality check
  /// for two values by or'ing the private bits together and stripping the sign. They are both zero,
  /// and therefore equivalent, if the resulting value is still zero.
  /// </summary>
  /// <param name="lhs">first value</param>
  /// <param name="rhs">second value</param>
  /// <returns>True if both arguments represent zero</returns>
  static bool AreZero(const BFloat16Impl& lhs, const BFloat16Impl& rhs) noexcept {
    // IEEE defines that positive and negative zero are equal, this gives us a quick equality check
    // for two values by or'ing the private bits together and stripping the sign. They are both zero,
    // and therefore equivalent, if the resulting value is still zero.
    return static_cast<uint16_t>((lhs.val | rhs.val) & ~kSignMask) == 0;
  }
};

template <class Derived>
inline uint16_t BFloat16Impl<Derived>::ToUint16Impl(float v) noexcept {
  uint16_t result;
  if (std::isnan(v)) {
    result = kPositiveQNaNBits;
  } else {
    auto get_msb_half = [](float fl) {
      uint16_t result;
#ifdef __cpp_if_constexpr
      if constexpr (detail::endian::native == detail::endian::little) {
#else
      if (detail::endian::native == detail::endian::little) {
#endif
        std::memcpy(&result, reinterpret_cast<char*>(&fl) + sizeof(uint16_t), sizeof(uint16_t));
      } else {
        std::memcpy(&result, &fl, sizeof(uint16_t));
      }
      return result;
    };

    uint16_t upper_bits = get_msb_half(v);
    union {
      uint32_t U32;
      float F32;
    };
    F32 = v;
    U32 += (upper_bits & 1) + kRoundToNearest;
    result = get_msb_half(F32);
  }
  return result;
}

template <class Derived>
inline float BFloat16Impl<Derived>::ToFloatImpl() const noexcept {
  if (IsNaN()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  float result;
  char* const first = reinterpret_cast<char*>(&result);
  char* const second = first + sizeof(uint16_t);
#ifdef __cpp_if_constexpr
  if constexpr (detail::endian::native == detail::endian::little) {
#else
  if (detail::endian::native == detail::endian::little) {
#endif
    std::memset(first, 0, sizeof(uint16_t));
    std::memcpy(second, &val, sizeof(uint16_t));
  } else {
    std::memcpy(first, &val, sizeof(uint16_t));
    std::memset(second, 0, sizeof(uint16_t));
  }
  return result;
}

}  // namespace onnxruntime_float16
