// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/version.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

using uninitialized_source = void*;
using Source =
    std::variant<uninitialized_source, std::reference_wrapper<std::istream>, std::reference_wrapper<const ov::Tensor>>;

class MetadataBase {
    class HelperStreambuf : std::streambuf {
    public:
        HelperStreambuf(const std::streambuf& other) {
            *this = other;
        }
        const char* data() {
            return gptr();
        }

        size_t get_byte_size() {
            return egptr() - gptr();
        }
    };

public:
    MetadataBase(uint32_t version, uint64_t blobDataSize);

    void read(const Source& source);

    virtual void read(const char* address, const size_t length) = 0;

    /**
     * @brief Writes metadata to a stream.
     */
    virtual void write(std::ostream& stream) = 0;

    virtual bool is_compatible() = 0;

    virtual uint64_t get_blob_size() const;

    /**
     * @returns The sizes of the init schedules. Populated only if "weights separation" has been enabled.
     */
    virtual std::optional<std::vector<uint64_t>> get_init_sizes() const;

    virtual std::optional<std::vector<ov::Layout>> get_input_layouts() const;

    virtual std::optional<std::vector<ov::Layout>> get_output_layouts() const;

    /**
     * @returns Batch size. Populated in case of plugin batching.
     */
    virtual std::optional<int64_t> get_batch_size() const;

    virtual ~MetadataBase() = default;

    static std::streampos getFileSize(std::istream& stream);

    virtual size_t get_metadata_size() const = 0;

    /**
     * @brief Returns a uint32_t value which represents two uint16_t values concatenated.
     * @details Convention for bumping the metadata version:
     *              - Increment Major in case of: removing a current field OR adding a new field in between fields.
     *              - Increment Minor in case of: adding a new field at the end.
     *
     * @return Major and minor versions concatenated into a single uint32_t value.
     */
    static constexpr uint32_t make_version(uint16_t major, uint16_t minor) {
        return major << 16 | (minor & 0x0000ffff);
    }

    /**
     * @brief Gets the major version.
     * @return Major version.
     */
    static constexpr uint16_t get_major(uint32_t version) {
        return static_cast<uint16_t>(version >> 16);
    }

    /**
     * @brief Gets the minor version.
     * @return Minor version.
     */
    static constexpr uint16_t get_minor(uint32_t version) {
        return static_cast<uint16_t>(version);
    }

protected:
    /**
     * @brief Reads data from the source containing the metadata. The implementation depends on the type of source.
     */
    void read_data_from_source(char* destination, const size_t size);

    /**
     * @brief Adds the size of the binary object and the magic string to the end of the stream.
     * @details This should be called after the "write" method in order to conclude writing the metadata into the given
     * stream.
     * @note This operation was detached from "write" since "write" writes at the beginning of the stream, while this
     * method writes at the end. This change allows better extension of class hierarchy.
     */
    void append_padding_blob_size_and_magic(std::ostream& stream);

    uint32_t _version;
    uint64_t _blobDataSize;
    Logger _logger;

    /**
     * @brief Where the metadata is read from. The type can be a stream, an OpenVINO tensor or "uninitialized_source".
     * @details Stored as attribute in order to avoid repeatedly passing the same arguments to some methods.
     * "uninitialized_source" (void*) is the default type assigned upon creation.
     */

    /**
     * @brief Used only when the source buffer is an OV tensor for managing the read coursor.
     */
    size_t _cursorOffset = 0;
};

/**
 * @brief Magic bytes used for identifying NPU blobs.
 */
constexpr std::string_view MAGIC_BYTES = "OVNPU";

/**
 * @brief List of supported version formats.
 */
constexpr uint32_t METADATA_VERSION_2_0{MetadataBase::make_version(2, 0)};
constexpr uint32_t METADATA_VERSION_2_1{MetadataBase::make_version(2, 1)};
constexpr uint32_t METADATA_VERSION_2_2{MetadataBase::make_version(2, 2)};
constexpr uint32_t METADATA_VERSION_2_3{MetadataBase::make_version(2, 3)};
constexpr uint32_t METADATA_VERSION_3_0{MetadataBase::make_version(3, 0)};

/**
 * @brief Current metadata version.
 */
constexpr uint32_t CURRENT_METADATA_VERSION{METADATA_VERSION_2_3};

constexpr uint16_t CURRENT_METADATA_MAJOR_VERSION{MetadataBase::get_major(CURRENT_METADATA_VERSION)};
constexpr uint16_t CURRENT_METADATA_MINOR_VERSION{MetadataBase::get_minor(CURRENT_METADATA_VERSION)};

class OpenvinoVersion final {
public:
    constexpr OpenvinoVersion(uint16_t major, uint16_t minor, uint16_t patch)
        : _major(major),
          _minor(minor),
          _patch(patch) {}

    OpenvinoVersion(const OpenvinoVersion& version);

    OpenvinoVersion& operator=(const OpenvinoVersion& other) {
        if (this != &other) {
            _major = other.get_major();
            _minor = other.get_minor();
            _patch = other.get_patch();
        }

        return *this;
    }

    ~OpenvinoVersion() = default;

    /**
     * @brief Reads version data from either an std::istream or a ov::Tensor.
     */
    void read(const char* address, const size_t length);

    /**
     * @brief Writes version data to a stream.
     */
    void write(std::ostream& stream);

    uint16_t get_major() const;

    uint16_t get_minor() const;

    uint16_t get_patch() const;

    size_t get_openvino_version_size() const;

    bool operator!=(const OpenvinoVersion& version);

private:
    uint16_t _major;
    uint16_t _minor;
    uint16_t _patch;
};

constexpr OpenvinoVersion CURRENT_OPENVINO_VERSION(OPENVINO_VERSION_MAJOR,
                                                   OPENVINO_VERSION_MINOR,
                                                   OPENVINO_VERSION_PATCH);

/**
 * @brief Template for metadata class handling.
 */
template <uint32_t version>
struct Metadata : public MetadataBase {};

/**
 * @brief Template specialization for metadata version 2.0.
 */
template <>
class Metadata<METADATA_VERSION_2_0> : public MetadataBase {
public:
    Metadata(uint64_t blobSize, const std::optional<OpenvinoVersion>& ovVersion = std::nullopt);

    void read(const char* address, const size_t length) override;

    /**
     * @attention It's a must to first write metadata version in any metadata specialization.
     *
     * @details When importing a versioned blob, it's best to first read the metadata version field.
     * This is the quickest way to handle many incompatible blob cases without needing to traverse the whole NPU
     * metadata section.
     */
    void write(std::ostream& stream) override;

    /**
     * @brief Checks if metadata is supported.
     *
     * @return Returns:
     *              - false:
     *                  - if blob metadata does not match current metadata.
     *                  - if blob OpenVINO version does not match current one.
     *
     *              - true: if all versions match.
     *
     * @note The version check can be disabled if the "OV_NPU_DISABLE_VERSION_CHECK" environment variable is set to
     * 'YES'.
     */
    bool is_compatible() override;

    size_t get_metadata_size() const override;

protected:
    OpenvinoVersion _ovVersion;
};

/**
 * @brief The version that adds support for init schedules (weights separation).
 */
template <>
class Metadata<METADATA_VERSION_2_1> : public Metadata<METADATA_VERSION_2_0> {
public:
    Metadata(uint64_t blobSize,
             const std::optional<OpenvinoVersion>& ovVersion = std::nullopt,
             const std::optional<std::vector<uint64_t>>& initSizes = std::nullopt);

    /**
     * @details The number of init schedules, along with the size of each init binary object are read in addition to the
     * information provided by the previous metadata versions.
     */
    void read(const char* address, const size_t length) override;

    /**
     * @details The number of init schedules, along with the size of each init binary object are written in addition to
     * the information registered by the previous metadata versions.
     */
    void write(std::ostream& stream) override;

    std::optional<std::vector<uint64_t>> get_init_sizes() const override;

    size_t get_metadata_size() const override;

private:
    std::optional<std::vector<uint64_t>> _initSizes;
    uint64_t _numberOfInits = 0;
};

/**
 * @brief The version that adds support for batch value storage.
 */
template <>
class Metadata<METADATA_VERSION_2_2> : public Metadata<METADATA_VERSION_2_1> {
public:
    Metadata(uint64_t blobSize,
             std::optional<OpenvinoVersion> ovVersion = std::nullopt,
             const std::optional<std::vector<uint64_t>> initSizes = std::nullopt,
             const std::optional<int64_t> batchSize = std::nullopt);

    void read(const char* address, const size_t length) override;

    void write(std::ostream& stream) override;

    std::optional<int64_t> get_batch_size() const override;

    size_t get_metadata_size() const override;

private:
    std::optional<int64_t> _batchSize;
};

/**
 * @brief Stores the layouts for all inputs and outputs (Parameter and Result nodes).
 * @details The order used for recording the layouts follows the deterministic order in which OV parses the I/O.
 */
template <>
class Metadata<METADATA_VERSION_2_3> : public Metadata<METADATA_VERSION_2_2> {
public:
    Metadata(uint64_t blobSize,
             const std::optional<OpenvinoVersion>& ovVersion = std::nullopt,
             const std::optional<std::vector<uint64_t>>& initSizes = std::nullopt,
             const std::optional<int64_t> batchSize = std::nullopt,
             const std::optional<std::vector<ov::Layout>>& inputLayouts = std::nullopt,
             const std::optional<std::vector<ov::Layout>>& outputLayouts = std::nullopt);

    void read(const char* address, const size_t length) override;

    void write(std::ostream& stream) override;

    size_t get_metadata_size() const override;

    std::optional<std::vector<ov::Layout>> get_input_layouts() const override;

    std::optional<std::vector<ov::Layout>> get_output_layouts() const override;

private:
    std::optional<std::vector<ov::Layout>> _inputLayouts;
    std::optional<std::vector<ov::Layout>> _outputLayouts;
};

/**
 * @brief Creates a Metadata object.
 *
 * @return Unique pointer to the created MetadataBase object if the major version is supported; otherwise, returns
 * 'nullptr'.
 */
std::unique_ptr<MetadataBase> create_metadata(uint32_t version, uint64_t blobSize);

/**
 * @brief Reads metadata from a blob (istream).
 *
 * @return If the blob is versioned and its major version is supported, returns an unique pointer to the read
 * MetadataBase object; otherwise, returns 'nullptr'.
 */
std::unique_ptr<MetadataBase> read_metadata_from(std::istream& stream);

/**
 * @brief Reads metadata from a blob (ov::Tensor).
 *
 * @return If the blob is versioned and its major version is supported, returns an unique pointer to the read
 * MetadataBase object; otherwise, returns 'nullptr'.
 */
std::unique_ptr<MetadataBase> read_metadata_from(const ov::Tensor& tensor);

/**
 * @brief Changes logic of fixed metadata sections to dynamic approach
 * Sections will no longer need to respect certain order and they will be parsed by their header containing type and
 * length In order to make a certain section mandatory, it needs to satisfy expression section
 */
template <>
class Metadata<METADATA_VERSION_3_0> : MetadataBase {
public:
    struct Section {
        enum class ESectionType : uint16_t {
            E_NPU_SECTION_EXPR = 0,
            E_NPU_WEIGHTLESS_BLOB,
            E_NPU_BATCH_SIZE,
            E_NPU_IO_LAYOUTS,
        };

        struct SectionHeader {
            uint16_t _type;
            uint16_t _length;
        };

        class ISectionBody {
        public:
            virtual uint16_t get_type() const = 0;
            virtual uint16_t get_byte_size() const = 0;
            virtual void parse_section(const char* addr) = 0;
            virtual void write_section(std::ostream& ostream) const = 0;
        };

        /* #define REGISTER_SECTION(SECTION_NAME)                      \
            class SECTION_NAME : ISectionBody {                      \
            public:                                                 \
                uint16_t get_type() override {                      \
                    return static_cast<uint16_t>(ESectionType::E_##SECTION_NAME);          \
                }                                                   \
                uint16_t get_byte_size() override;                  \
                void parse_section(const char* addr) override;   \
                void write_section(std::ostream& ostream) const override; \
            }

            REGISTER_SECTION(NPU_SECTION_EXPR);
            REGISTER_SECTION(NPU_WEIGHTLESS_BLOB);
            REGISTER_SECTION(NPU_BATCH_SIZE);
            REGISTER_SECTION(NPU_IO_LAYOUTS);

        #undef REGISTER_SECTION */

        class NPU_SECTION_EXPR : public ISectionBody {
            enum class EOperators : uint32_t {
                E_AND = std::numeric_limits<uint16_t>::max() + 1,  // 65536
                E_OR,
                E_PARENTHESES,
                E_NOT_FOUND
            };

        public:
            uint16_t get_type() const override {
                return static_cast<uint16_t>(ESectionType::E_NPU_SECTION_EXPR);
            }

            uint16_t get_byte_size() const override {
                return sizeof(uint64_t) + _expression.size() * sizeof(decltype(_expression)::value_type);
            }

            void parse_section(const char* addr) override {
                uint64_t expressionSize = *reinterpret_cast<const uint64_t*>(addr);
                _expression.resize(expressionSize);
                std::memcpy(static_cast<void*>(_expression.data()),
                            addr + sizeof(expressionSize),
                            expressionSize * sizeof(decltype(_expression)::value_type));
            }

            void write_section(std::ostream& ostream) const override {
                uint64_t expressionSize = _expression.size();
                ostream.write(reinterpret_cast<const char*>(&expressionSize), sizeof(expressionSize));
                ostream.write(reinterpret_cast<const char*>(_expression.data()),
                              _expression.size() * sizeof(decltype(_expression)::value_type));
            }

            bool evaluate(const std::vector<Section>& supportedSections) const {
                /* TO DO */
                auto currentOperator = parse_operator(*_expression.cbegin());
                OPENVINO_ASSERT(currentOperator != EOperators::E_NOT_FOUND,
                                "Expression should start with an operator!");
                for (auto it = std::next(_expression.cbegin(), 1); it != _expression.cend(); it++) {
                    auto operatorCheck = parse_operator(*it);
                    if (operatorCheck != EOperators::E_NOT_FOUND) {
                        currentOperator = operatorCheck;
                    } else if (currentOperator == EOperators::E_AND && std::find_if(supportedSections.cbegin(),
                                                                                    supportedSections.cend(),
                                                                                    [&it](const Section& section) {
                                                                                        if (*it == section.get_type()) {
                                                                                            return true;
                                                                                        }
                                                                                        return false;
                                                                                    }) == supportedSections.cend()) {
                        return false;
                    }
                }
                return true;
            }

        private:
            EOperators parse_operator(const uint32_t val) const {
                switch (val) {
                case static_cast<uint32_t>(EOperators::E_AND):
                    return EOperators::E_AND;
                case static_cast<uint32_t>(EOperators::E_OR):
                    return EOperators::E_OR;
                case static_cast<uint32_t>(EOperators::E_PARENTHESES):
                    return EOperators::E_PARENTHESES;
                default:
                    return EOperators::E_NOT_FOUND;
                }
                OPENVINO_THROW("Unreacheable code reached!");
            }
            std::vector<uint32_t> _expression;
        };

        class NPU_WEIGHTLESS_BLOB : public ISectionBody {
        public:
            uint16_t get_type() const override {
                return static_cast<uint16_t>(ESectionType::E_NPU_WEIGHTLESS_BLOB);
            }

            uint16_t get_byte_size() const override {
                return sizeof(uint64_t) + _initSizes.size() * sizeof(decltype(_initSizes)::value_type);
            }

            void parse_section(const char* addr) override {
                uint64_t numberOfInits = *reinterpret_cast<const uint64_t*>(addr);
                _initSizes.resize(numberOfInits);
                std::memcpy(static_cast<void*>(_initSizes.data()),
                            addr + sizeof(numberOfInits),
                            numberOfInits * sizeof(uint32_t));
            }

            void write_section(std::ostream& ostream) const override {
                uint64_t numberOfInits = _initSizes.size();
                ostream.write(reinterpret_cast<const char*>(&numberOfInits), sizeof(numberOfInits));
                ostream.write(reinterpret_cast<const char*>(_initSizes.data()),
                              _initSizes.size() * sizeof(decltype(_initSizes)::value_type));
            }

        private:
            std::vector<uint64_t> _initSizes;
        };

        class NPU_BATCH_SIZE : public ISectionBody {
        public:
            uint16_t get_type() const override {
                return static_cast<uint16_t>(ESectionType::E_NPU_BATCH_SIZE);
            }

            uint16_t get_byte_size() const override {
                return sizeof(_batchSize);
            }

            void parse_section(const char* addr) override {
                _batchSize = *reinterpret_cast<const int64_t*>(addr);
            }

            void write_section(std::ostream& ostream) const override {
                ostream.write(reinterpret_cast<const char*>(_batchSize), sizeof(_batchSize));
            }

        private:
            int64_t _batchSize;
        };

        class NPU_IO_LAYOUTS : public ISectionBody {
        public:
            uint16_t get_type() const override {
                return static_cast<uint16_t>(ESectionType::E_NPU_IO_LAYOUTS);
            }

            uint16_t get_byte_size() const override {
                uint64_t totalSize = 2 * sizeof(uint64_t);
                for (const auto& inputLayout : _inputLayouts) {
                    const std::string layoutString = inputLayout.to_string();
                    const uint16_t stringLength = static_cast<uint16_t>(layoutString.size());
                    totalSize += sizeof(uint16_t) + stringLength;
                }

                for (const auto& outputLayout : _outputLayouts) {
                    const std::string layoutString = outputLayout.to_string();
                    const uint16_t stringLength = static_cast<uint16_t>(layoutString.size());
                    totalSize += sizeof(uint16_t) + stringLength;
                }
                return totalSize;
            }

            void parse_section(const char* addr) override {
                const uint64_t numberOfInputLayouts = *reinterpret_cast<const uint64_t*>(addr);
                uint64_t offset = sizeof(numberOfInputLayouts);
                const uint64_t numberOfOutputLayouts = *reinterpret_cast<const uint64_t*>(addr + offset);
                offset += sizeof(numberOfInputLayouts);
                for (uint64_t i = 0; i < numberOfInputLayouts; ++i) {
                    const uint16_t stringlength = *reinterpret_cast<const uint16_t*>(addr + offset);
                    offset += sizeof(stringlength);
                    _inputLayouts.push_back(ov::Layout(std::string(addr + offset, stringlength)));
                    offset += stringlength;
                }
                for (uint64_t i = 0; i < numberOfOutputLayouts; ++i) {
                    const uint16_t stringlength = *reinterpret_cast<const uint16_t*>(addr + offset);
                    offset += sizeof(stringlength);
                    _outputLayouts.push_back(ov::Layout(std::string(addr + offset, stringlength)));
                    offset += stringlength;
                }
            }

            void write_section(std::ostream& ostream) const override {
                const uint64_t numberOfInputLayouts = _inputLayouts.size();
                const uint64_t numberOfOutputLayouts = _outputLayouts.size();
                ostream.write(reinterpret_cast<const char*>(&numberOfInputLayouts), sizeof(numberOfInputLayouts));
                ostream.write(reinterpret_cast<const char*>(&numberOfOutputLayouts), sizeof(numberOfOutputLayouts));
                for (const auto& intputLayout : _inputLayouts) {
                    auto layoutStr = intputLayout.to_string();
                    const uint16_t layoutStrSize = layoutStr.size();
                    ostream.write(reinterpret_cast<const char*>(&layoutStrSize), sizeof(layoutStrSize));
                    ostream.write(layoutStr.data(), layoutStr.size());
                }
                for (const auto& intputLayout : _outputLayouts) {
                    auto layoutStr = intputLayout.to_string();
                    const uint16_t layoutStrSize = layoutStr.size();
                    ostream.write(reinterpret_cast<const char*>(&layoutStrSize), sizeof(layoutStrSize));
                    ostream.write(layoutStr.data(), layoutStr.size());
                }
            }

        private:
            std::vector<ov::Layout> _inputLayouts;
            std::vector<ov::Layout> _outputLayouts;
        };

        Section(const char* address, size_t& offset) {
            _sHead = *reinterpret_cast<const SectionHeader*>(address);
            offset += sizeof(SectionHeader);
            _sBodyOffset = address + offset;
            offset += _sHead._length;
            _sBodyPtr = nullptr;
        }

        Section(const std::shared_ptr<ISectionBody>& ISectionBody) {
            _sHead._type = ISectionBody->get_type();
            _sHead._length = ISectionBody->get_byte_size();
            _sBodyPtr = ISectionBody;
        }

        size_t get_type() const {
            return _sHead._type;
        }

        size_t get_length() const {
            return sizeof(SectionHeader) + _sHead._length;
        }

        void write_section(std::ostream& stream) const {
            stream.write(reinterpret_cast<const char*>(&_sHead), sizeof(SectionHeader));
            _sBodyPtr->write_section(stream);
        }

        SectionHeader _sHead;
        std::shared_ptr<ISectionBody> _sBodyPtr = nullptr;
        const char* _sBodyOffset = nullptr;
    };

    void read(const char* address, const size_t length) override {
        parse_sections(address, length);
    }

    void write(std::ostream& stream) override {
        write_sections(stream);
    }

private:
    void append_section(const std::shared_ptr<Section::ISectionBody>& ISectionBody) {
        _sections.push_back(Section(ISectionBody));
    }

    void parse_sections(const char* address, const size_t length) {
        size_t offset = 0;
        do {
            auto section = Section(address, offset);
            _sections.push_back(section);
            offset += section.get_length();
        } while (offset < length);
    }

    void write_sections(std::ostream& ostream) const {
        for (const auto& section : _sections) {
            section.write_section(ostream);
        }
    }

    std::vector<Section> _sections;
};

}  // namespace intel_npu
