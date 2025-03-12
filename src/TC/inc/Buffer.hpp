/*
 * Copyright 2025 Vision Labs LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <memory>

namespace VPF {
/**
 * PyArrayInterface structure is defined in numpy header which isn't
 * available outside the whole numpy repo.
 *
 * So this structure stores values to be parsed from dictionary with
 * Python specification of PyArrayInterface.
 */
struct PyArrayInterfaceDescriptor {
  /** We are only interested in using arrays as Host-side Surface
   * representation. Hence no need for more then 3 dimensions.
   **/
  static constexpr size_t m_num_elems = 3U;

  // A string providing the basic type of the homogeneous array.
  std::string m_typestr = "V";

  // Pointer to the first element of data and read-only flag.
  void* m_ptr = nullptr;
  bool m_read_only = false;

  // Size of each dimension.
  unsigned int m_shape[m_num_elems] = {};

  // Strides (distance between 2 vertically adjacent pixels in bytes).
  unsigned int m_strides[m_num_elems] = {};

  // Spec version, 3 for now.
  int const m_version = 3;
};
}; // namespace VPF

/* Represents CPU-side memory.
 * May own the memory or be a wrapper around existing ponter;
 */
class TC_EXPORT Buffer final : public Token {
public:
  Buffer() = delete;
  Buffer(const Buffer& other) = delete;
  Buffer& operator=(Buffer& other) = delete;

  ~Buffer() final;

  /// @brief Check if Buffer is created from PAI.
  /// @return true if created from PAI, false otherwise.
  bool FromPAI() const;

  /// @brief Get Python Array Interface descriptor.
  /// @return const reference to descriptor.
  const PyArrayInterfaceDescriptor& GetPAIDescr () const;

  /// @brief Get raw data pointer
  /// @return pointer to first byte
  void* GetRawMemPtr();

  /// @brief Get raw const data pointer
  /// @return pointer to first byte
  const void* GetRawMemPtr() const;

  /// @brief Get amount of allocated memory in bytes
  /// @return size in bytes
  size_t GetRawMemSize() const;

  /// @brief Resize and copy from pointer
  /// @param newSize new size in bytes
  /// @param newPtr pointer to copy from
  void Update(size_t newSize, void* newPtr = nullptr);

  /// @brief Copy from pointer
  /// @param size Amount of bytes to copy
  /// @param ptr pointer to copy from
  /// @retval true in case of success
  /// @retval false if size != buffer size
  /// @retval false if memory isn't allocated
  bool CopyFrom(size_t size, void const* ptr);

  /// @brief Get pointer as pointer to specific type
  /// @tparam T type
  /// @return typed pointer
  template <typename T> T* GetDataAs() { return (T*)GetRawMemPtr(); }

  /// @brief Get const pointer as pointer to specific type
  /// @tparam T type
  /// @return typed pointer
  template <typename T> T const* GetDataAs() const {
    return (T const*)GetRawMemPtr();
  }

  /// @brief Don't own memory, don't allocate, only store size.
  /// @param bufferSize size in bytes
  /// @return
  static Buffer* Make(size_t bufferSize);

  /// @brief Don't own, just a memory view
  /// @param bufferSize buffer size
  /// @param pCopyFrom pointer to allocated memory
  /// @return
  static Buffer* Make(size_t bufferSize, void* pCopyFrom);

  /// @brief Don't own memory, just a view of Array Interface
  /// @param descr descriptor
  /// @return
  static Buffer* Make(const PyArrayInterfaceDescriptor& descr);

  /// @brief Own memory, allocate
  /// @param bufferSize size in bytes to be allocated
  /// @return
  static Buffer* MakeOwnMem(size_t bufferSize);

  /// @brief Own memory, allocate, deep copy
  /// @param bufferSize size in bytes to be allocated
  /// @param pCopyFrom pointer to copy from
  /// @return
  static Buffer* MakeOwnMem(size_t bufferSize, const void* pCopyFrom);

private:
  explicit Buffer(size_t bufferSize, bool ownMemory = true);

  Buffer(size_t bufferSize, void* pCopyFrom, bool ownMemory);

  Buffer(size_t bufferSize, const void* pCopyFrom);

  /// @brief The actual allocation will happen here if buffer owns the memory
  /// @return true in case of success, false otherwise
  bool Allocate();

  /// @brief Deallocation will happen here if buffer owns the memory
  void Deallocate();

  bool m_own_memory = true;

  size_t m_mem_size = 0UL;

  void* m_p_raw_data = nullptr;

#ifdef TRACK_TOKEN_ALLOCATIONS
  uint32_t id;
#endif

  struct PyArrayInterfaceContext {
    /// @brief true if Buffer is created over array interface, false otherwise.
    bool m_present;

    /// @brief Descriptor.
    VPF::PyArrayInterfaceDescriptor m_descr;
  } m_pai_ctx;
};