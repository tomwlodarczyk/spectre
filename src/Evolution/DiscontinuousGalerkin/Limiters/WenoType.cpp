// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Limiters/WenoType.hpp"

#include <ostream>
#include <string>

#include "ErrorHandling/Error.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"

std::ostream& Limiters::operator<<(
    std::ostream& os, const Limiters::WenoType weno_type) noexcept {
  switch (weno_type) {
    case Limiters::WenoType::Hweno:
      return os << "Hweno";
    case Limiters::WenoType::SimpleWeno:
      return os << "SimpleWeno";
    default:  // LCOV_EXCL_LINE
      // LCOV_EXCL_START
      ERROR("Missing a case for operator<<(WenoType)");
      // LCOV_EXCL_STOP
  }
}

template <>
Limiters::WenoType create_from_yaml<Limiters::WenoType>::create<void>(
    const Option& options) {
  const std::string weno_type_read = options.parse_as<std::string>();
  if (weno_type_read == "Hweno") {
    return Limiters::WenoType::Hweno;
  } else if (weno_type_read == "SimpleWeno") {
    return Limiters::WenoType::SimpleWeno;
  }
  PARSE_ERROR(options.context(), "Failed to convert \""
                                     << weno_type_read
                                     << "\" to WenoType. Expected one of: "
                                        "{Hweno, SimpleWeno}.");
}
