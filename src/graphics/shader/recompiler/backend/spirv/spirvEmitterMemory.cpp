#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <algorithm>
#include <bit>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t AndCondition(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return Binary(state, spv::OpLogicalAnd, TypeBool(state), lhs, rhs);
}

uint32_t EmitDsMaskedLaneRead(EmitterState& state, uint32_t source, uint32_t target,
                              uint32_t exec) {
	if (state.lane_count == 2) {
		target = Binary(state, spv::OpBitwiseAnd, TypeU32(state), target, ConstantU32(state, 31));
	}
	const auto shuffled = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), shuffled,
	                          ConstantU32(state, spv::ScopeSubgroup), source, target);
	const auto source_exec = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeBool(state), source_exec,
	                          ConstantU32(state, spv::ScopeSubgroup), exec, target);
	const auto source_active =
	    AndCondition(state, source_exec, EmitSubgroupLaneActiveBool(state, target));
	return Select(state, TypeU32(state), source_active, shuffled, ConstantU32(state, 0));
}

struct BufferAddress {
	uint32_t offset;
	uint32_t byte;
};

BufferAddress CalculateBufferAddress(EmitterState& state, uint32_t index, uint32_t offset,
                                     uint32_t soffset, uint32_t immediate, uint32_t stride,
                                     uint32_t swizzle, uint32_t index_stride) {
	const auto zero = ConstantU32(state, 0);
	const auto one  = ConstantU32(state, 1);
	const auto add = [&](uint32_t lhs, uint32_t rhs) {
		return lhs == zero ? rhs : rhs == zero ? lhs
		                                      : Binary(state, spv::OpIAdd, TypeU32(state), lhs, rhs);
	};
	const auto mul = [&](uint32_t lhs, uint32_t rhs) {
		return lhs == zero || rhs == zero ? zero
		       : lhs == one              ? rhs
		       : rhs == one              ? lhs
		                                 : Binary(state, spv::OpIMul, TypeU32(state), lhs, rhs);
	};
	if (immediate != 0u) {
		offset = add(offset, ConstantU32(state, immediate));
	}
	auto address = add(mul(index, stride), offset);
	if (swizzle != 0u) {
		const auto index_shift =
		    Binary(state, spv::OpIAdd, TypeU32(state), index_stride, ConstantU32(state, 3));
		const auto indices = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		                            ConstantU32(state, 1), index_shift);
		const auto index_msb =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), index, index_shift);
		const auto index_lsb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), index,
		           Binary(state, spv::OpISub, TypeU32(state), indices, ConstantU32(state, 1)));
		const auto offset_msb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, ~3u));
		const auto offset_lsb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, 3u));
		const auto msb = mul(add(mul(index_msb, stride), offset_msb), indices);
		const auto lsb = add(Binary(state, spv::OpShiftLeftLogical, TypeU32(state), index_lsb,
		                            ConstantU32(state, 2u)), offset_lsb);
		address = Select(state, TypeU32(state), swizzle, add(msb, lsb), address);
	}
	return {offset, add(address, soffset)};
}

uint32_t BufferLane(EmitterState& state) {
	return Binary(state, spv::OpBitwiseAnd, TypeU32(state), EmitSubgroupLocalInvocationId(state),
	              ConstantU32(state, 63));
}

uint32_t BufferByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                           uint32_t dynamic_stride = 0) {
	auto&      state  = ctx.state;
	const auto packed = StorageBufferPackedStride(state, mem);
	const auto stride = packed & 0x3fffu;
	auto       index  = ctx.Arg(inst, 1);
	if ((packed & (1u << 20u)) != 0u) {
		index = Binary(state, spv::OpIAdd, TypeU32(state), index, BufferLane(state));
	}
	const bool swizzle   = (dynamic_stride != 0u || stride != 0u) && (packed & (1u << 14u)) != 0u;
	const auto stride_id = dynamic_stride != 0u ? dynamic_stride : ConstantU32(state, stride);
	return CalculateBufferAddress(state, index, ctx.Arg(inst, 2), ctx.Arg(inst, 3), mem.offset,
	                              stride_id,
	                              swizzle ? ConstantBool(state, true) : 0u,
	                              ConstantU32(state, (packed >> 16u) & 3u))
	    .byte;
}

uint32_t AddU64Low(EmitterState& state, uint32_t low, uint32_t high, uint32_t add_low,
                   uint32_t add_high, uint32_t& out_high) {
	const auto result = Binary(state, spv::OpIAdd, TypeU32(state), low, add_low);
	const auto carry  = Binary(state, spv::OpULessThan, TypeBool(state), result, low);
	out_high =
	    Binary(state, spv::OpIAdd, TypeU32(state),
	           Binary(state, spv::OpIAdd, TypeU32(state), high, add_high),
	           Select(state, TypeU32(state), carry, ConstantU32(state, 1), ConstantU32(state, 0)));
	return result;
}

uint32_t AddAddressOffset(EmitterState& state, const IR::MemoryInfo& mem, uint32_t low,
                          uint32_t& high) {
	const auto immediate = static_cast<int32_t>(mem.offset);
	if (immediate == 0) return low;
	const auto immediate_low  = ConstantU32(state, static_cast<uint32_t>(immediate));
	const auto immediate_high = ConstantU32(state, immediate < 0 ? UINT32_MAX : 0u);
	return AddU64Low(state, low, high, immediate_low, immediate_high, high);
}

uint32_t ScratchByteAddress(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t low,
                            uint32_t high) {
	auto& state = ctx.state;
	low = AddAddressOffset(state, mem, low, high);
	const auto valid = Binary(state, spv::OpIEqual, TypeBool(state), high, ConstantU32(state, 0));
	return Select(state, TypeU32(state), valid, low, ConstantU32(state, UINT32_MAX));
}

uint32_t GuestAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto& state = ctx.state;
	auto  low   = ctx.Arg(inst, 1);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, ~3u));
	}
	uint32_t address = 0;
	if (mem.address_is_full) {
		address = PackU64(state, low, ctx.Arg(inst, 2));
	} else {
		const auto* handle = inst.Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != IR::ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) {
			ctx.Fail(inst, "has no address base pair");
			return ConstantU64(state, 0);
		}
		auto base_low = ctx.Arg(*handle, 0);
		if (mem.kind == IR::ResourceKind::ScalarAddress) {
			base_low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), base_low,
			                  ConstantU32(state, ~3u));
		}
		const auto base = PackU64(state, base_low, ctx.Arg(*handle, 1));
		address         = Binary(state, spv::OpIAdd, TypeU64(state), base,
		                         Unary(state, spv::OpUConvert, TypeU64(state), low));
	}
	auto immediate = static_cast<int32_t>(mem.offset);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		immediate = static_cast<int32_t>(static_cast<uint32_t>(immediate) & ~3u);
	}
	return immediate == 0
	           ? address
	           : Binary(state, spv::OpIAdd, TypeU64(state), address,
	                    ConstantU64(state, static_cast<uint64_t>(static_cast<int64_t>(immediate))));
}

uint32_t FaultElementPointer(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.fault_buffer_variable, ConstantU32(state, 0), index);
	return pointer;
}

void RecordBdaFault(EmitterState& state, uint32_t page) {
	const auto word =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), page, ConstantU32(state, 5));
	const auto bit =
	    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), page, ConstantU32(state, 31)));
	// One atomic OR: lanes faulting on pages of the same bitmap word keep each other's bits.
	state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), state.builder.AllocateId(),
	                          FaultElementPointer(state, word),
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone), bit);
}

struct BdaPageLookup {
	uint32_t page;
	uint32_t entry;
	uint32_t bda;
	uint32_t present;
};

BdaPageLookup LookupBdaPage(EmitterState& state, uint32_t address) {
	const auto type     = TypeU64(state);
	const auto extended = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), address,
	                             ConstantU64(state, LibKernel::Memory::kExtendedMemoryBase));
	const auto packed   = Select(
	    state, type, extended,
	    Binary(state, spv::OpISub, type, address,
	           ConstantU64(state, LibKernel::Memory::kExtendedMemoryBase - LOWER_ADDRESS_SIZE)),
	    address);
	const auto page64 = Binary(state, spv::OpShiftRightLogical, type, packed,
	                           ConstantU32(state, BufferCache::CACHING_PAGEBITS));
	const auto page   = Unary(state, spv::OpUConvert, TypeU32(state), page64);
	const auto entry_pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state, 64),
	                          entry_pointer, state.bda_pagetable_variable, ConstantU32(state, 0),
	                          page);
	const auto entry = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, type, entry, entry_pointer);
	const auto present =
	    Binary(state, spv::OpINotEqual, TypeBool(state), entry, ConstantU64(state, 0));
	const auto offset = Binary(state, spv::OpBitwiseAnd, type, address,
	                           ConstantU64(state, BufferCache::CACHING_PAGESIZE - 1));
	const auto bda = Select(state, type, present, Binary(state, spv::OpIAdd, type, entry, offset),
	                        ConstantU64(state, 0));
	return {page, entry, bda, present};
}

void RecordMaskedBdaFault(EmitterState& state, uint32_t page, uint32_t unbacked) {
	const auto word =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), page, ConstantU32(state, 5));
	const auto raw_bit =
	    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), page, ConstantU32(state, 31)));
	const auto bit = Select(state, TypeU32(state), unbacked, raw_bit, ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), state.builder.AllocateId(),
	                          FaultElementPointer(state, word),
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone), bit);
}

uint32_t LoadBdaDwordFromPointer(EmitterState& state, uint32_t bda, bool coherent = false) {
	auto pointer = state.builder.AllocateId();
	if (coherent) {
		const auto block = state.builder.DecoratedType(
		    spv::OpTypeStruct, {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
		                        {spv::OpMemberDecorate, {0, spv::DecorationCoherent}},
		                        {spv::OpDecorate, {spv::DecorationBlock}}}, TypeU32(state));
		state.builder.AddFunction(
		    spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, block),
		    pointer, bda);
		const auto base = pointer;
		pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypePhysicalU32Pointer(state), pointer,
		                          base, ConstantU32(state, 0));
	} else {
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
		                          bda);
	}
	const auto         value     = state.builder.AllocateId();
	constexpr uint32_t alignment = sizeof(uint32_t);
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
	                          spv::MemoryAccessAlignedMask |
	                              (coherent ? spv::MemoryAccessVolatileMask : spv::MemoryAccessMaskNone),
	                          alignment);
	return value;
}

uint32_t LoadBdaDword(ValueEmitContext& ctx, uint32_t address, bool coherent = false) {
	auto&      state   = ctx.state;
	const auto bda     = GetBdaPointer(state, address);
	const auto present =
	    Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
	// A page that no buffer backs yet reads as zero, except in BVH traversal shaders: there it
	// reads as all ones, the absent-node link bvh_intersect returns, because guest list walks end
	// on negative links and would advance over zero entries forever.
	const auto missing = ConstantU32(state, state.program.info.bvh ? ~0u : 0u);
	return EmitValueOrDefaultIfCondition(state, present, TypeU32(state), missing, [&]() {
		return LoadBdaDwordFromPointer(state, bda, coherent);
	});
}

uint32_t LoadBda(ValueEmitContext& ctx, uint32_t address, uint32_t active, uint32_t bits,
                 bool coherent = false) {
	auto& state = ctx.state;
	return EmitValueOrZeroIfCondition(state, active, [&]() {
		const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), address,
		                            ConstantU64(state, ~uint64_t {3}));
		const auto first   = LoadBdaDword(ctx, aligned, coherent);
		const auto byte =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		           Unary(state, spv::OpUConvert, TypeU32(state), address), ConstantU32(state, 3));
		const auto crosses =
		    bits == 8u ? ConstantBool(state, false)
		               : Binary(state, bits == 16u ? spv::OpUGreaterThan : spv::OpINotEqual,
		                        TypeBool(state), byte, ConstantU32(state, bits == 16u ? 2u : 0u));
		const auto second = EmitValueOrZeroIfCondition(state, crosses, [&]() {
			return LoadBdaDword(ctx, Binary(state, spv::OpIAdd, TypeU64(state), aligned,
			                                ConstantU64(state, sizeof(uint32_t))), coherent);
		});
		const auto shift =
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), byte, ConstantU32(state, 3));
		const auto upper_shift =
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		           Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                  Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 4), byte),
		                  ConstantU32(state, 3)),
		           ConstantU32(state, 3));
		const auto merged =
		    Binary(state, spv::OpBitwiseOr, TypeU32(state),
		           Binary(state, spv::OpShiftRightLogical, TypeU32(state), first, shift),
		           Binary(state, spv::OpShiftLeftLogical, TypeU32(state), second, upper_shift));
		return bits == 32u ? merged
		                   : Binary(state, spv::OpBitwiseAnd, TypeU32(state), merged,
		                            ConstantU32(state, bits == 8u ? 0xffu : 0xffffu));
	});
}

uint32_t ByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                     uint32_t prefix = 0, uint32_t dynamic_stride = 0) {
	if (mem.kind == IR::ResourceKind::Buffer) {
		const auto resource = ctx.state.program.info.buffers.at(mem.resource).descriptor_index;
		if (prefix == 0u) prefix = ctx.state.memory_byte_offsets[resource];
		const auto address = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state),
		                            BufferByteAddress(ctx, inst, mem, dynamic_stride), prefix);
		// The host binding prefix must not wrap an out-of-range guest address into the buffer.
		return Select(ctx.state, TypeU32(ctx.state),
		              Binary(ctx.state, spv::OpUGreaterThanEqual, TypeBool(ctx.state), address, prefix),
		              address, ConstantU32(ctx.state, UINT32_MAX));
	}
	if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
		if (mem.offset == 0u) {
			return ctx.Arg(inst, 0);
		}
		return Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), ctx.Arg(inst, 0),
		              ConstantU32(ctx.state, mem.offset));
	}
	if (mem.kind != IR::ResourceKind::Scratch) {
		EXIT("physical address memory must use the BDA emitter\n");
	}
	return ScratchByteAddress(ctx, mem, ctx.Arg(inst, 1), ctx.Arg(inst, 2));
}

uint32_t DwordIndex(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                    uint32_t prefix = 0, uint32_t dynamic_stride = 0) {
	return Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state),
	              ByteAddress(ctx, inst, mem, prefix, dynamic_stride), ConstantU32(ctx.state, 2));
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index);

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits);

template <typename Fn>
auto EmitBufferAccess(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem,
                      uint32_t components, Fn&& emit) {
	auto& state = ctx.state;
	constexpr bool has_result = !std::is_void_v<
	    std::invoke_result_t<Fn, const IR::MemoryInfo&, const MemoryResourceAccess&>>;
	const auto access = [&](uint32_t resource, uint32_t array_index = 0u) {
		mem.resource = resource;
		if (mem.kind == IR::ResourceKind::Buffer &&
		    IsEmptyIndirectBufferCandidate(state, resource)) {
			// Empty candidates still carry the format/swizzle needed for OOB loads.
			return emit(mem, MemoryResourceAccess{});
		}
		return emit(mem, PrepareMemoryResourceAccess(state, mem, array_index));
	};
	if (mem.kind != IR::ResourceKind::Buffer ||
	    state.program.info.buffers[mem.resource].indirect_root != mem.resource) {
		return access(mem.resource);
	}
	const auto root_resource = mem.resource;
	const auto& buffer = state.program.info.buffers[root_resource];
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto selected = ctx.Arg(handle, 0);
	const bool dynamic_buffers = state.program.bindings.dynamic_indirect_buffers;
	const auto representative =
	    dynamic_buffers ? UINT32_MAX : SharedIndirectBufferResource(state, root_resource);
	const auto shared = representative != UINT32_MAX;
	const auto empty_root = shared && buffer.packed_stride == 0u;
	uint32_t valid_bound = static_cast<uint32_t>(buffer.indirect_resources.size());
	const auto plan_it = state.indirect_plans.find(&inst);
	if (dynamic_buffers && plan_it != state.indirect_plans.end() &&
	    plan_it->second.map_capacity != 0u) {
		valid_bound = plan_it->second.map_capacity;
	}
	const auto selected_access = [&]() {
		if (shared) {
			const auto root_slot = buffer.descriptor_index;
			const auto child_slot = state.program.info.buffers[buffer.indirect_resources[1]].descriptor_index;
			auto slot = EmitAddU32(state, selected, ConstantU32(state, child_slot - 1u));
			if (!empty_root && child_slot != root_slot + 1u) {
				slot = Select(state, TypeU32(state),
				    Binary(state, spv::OpIEqual, TypeBool(state), selected, ConstantU32(state, 0u)),
				    ConstantU32(state, root_slot), slot);
			}
			mem.resource = representative;
			return emit(mem, PrepareMemoryResourceAccess(state, mem, slot));
		}
		if (buffer.indirect_resources.empty()) {
			if constexpr (has_result) {
				return ConstantU32CompositeZero(state, components);
			} else {
				return;
			}
		}
		IndirectResourcePlan fallback_plan;
		if (plan_it == state.indirect_plans.end()) {
			fallback_plan = BuildIndirectBufferPlan(state, mem);
		}
		const auto& plan = plan_it != state.indirect_plans.end() ? plan_it->second : fallback_plan;
		const auto [group_index, slot] = SelectIndirectResourceGroup(state, plan, selected);
		const auto EmitGroup = [&](uint32_t index) {
			const auto& group = plan.groups[index];
			return access(group.resource, group.needs_slot ? slot : 0u);
		};
		return plan.groups.size() == 1u
		           ? EmitGroup(0u)
		           : EmitIndexSwitch(state, group_index,
		                             static_cast<uint32_t>(plan.groups.size()),
		                             has_result ? TypeU32Composite(state, components) : 0u,
		                             EmitGroup);
	};
	auto valid = Binary(state, spv::OpULessThan, TypeBool(state), selected,
	                    ConstantU32(state, valid_bound));
	if constexpr (has_result) {
		return EmitValueOrDefaultIfCondition(state, valid, TypeU32Composite(state, components),
		    ConstantU32CompositeZero(state, components), [&]() {
			    if (!empty_root) return selected_access();
			    return EmitValueOrDefaultIfCondition(state,
			        Binary(state, spv::OpINotEqual, TypeBool(state), selected, ConstantU32(state, 0u)),
			        TypeU32Composite(state, components), access(root_resource), selected_access);
		    });
	} else {
		if (empty_root) valid = AndCondition(state, valid,
		    Binary(state, spv::OpINotEqual, TypeBool(state), selected, ConstantU32(state, 0u)));
		EmitIfCondition(state, valid, selected_access);
	}
}

uint32_t BufferIndexInBounds(ValueEmitContext& ctx, const IR::Inst& inst,
                             const IR::MemoryInfo& mem, const MemoryResourceAccess& resource,
                             uint32_t payload_in_bounds) {
	if (mem.kind != IR::ResourceKind::Buffer) return payload_in_bounds;
	auto& state = ctx.state;
	const auto& buffer = state.program.info.buffers[mem.resource];
	const auto& sources = state.program.descriptor_sources;
	if (buffer.source >= sources.size() || !sources[buffer.source].indirect_descriptor ||
	    !sources[buffer.source].indirect_descriptor->indexed) return payload_in_bounds;
	const auto bytes = Binary(state, spv::OpIMul, TypeU32(state), resource.length,
	                          ConstantU32(state, resource.element_bits / 8u));
	const auto first_byte = mem.offset == 0u ? resource.byte_offset
	    : EmitAddU32(state, resource.byte_offset, ConstantU32(state, mem.offset));
	const auto available = Binary(state, spv::OpISub, TypeU32(state), bytes, first_byte);
	// Bound the original index before multiplying by the stride and adding the field offset.
	const auto stride_id = resource.stride != 0u
	                           ? EmitUMax32(state, resource.stride, ConstantU32(state, 1u))
	                           : ConstantU32(state, buffer.packed_stride & 0x3fffu);
	const auto last_record = Binary(state, spv::OpUDiv, TypeU32(state),
	    Binary(state, spv::OpISub, TypeU32(state), available, ConstantU32(state, 1u)),
	    stride_id);
	return AndCondition(state, payload_in_bounds,
	    AndCondition(state,
	        Binary(state, spv::OpUGreaterThan, TypeBool(state), bytes, first_byte),
	        Binary(state, spv::OpULessThanEqual, TypeBool(state), ctx.Arg(inst, 1), last_record)));
}

uint32_t LoadWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource) {
	const auto index = DwordIndex(ctx, inst, mem, resource.byte_offset, resource.stride);
	return EmitValueOrZeroIfCondition(
	    ctx.state, EmitStorageElementInBounds(ctx.state, resource, index),
	    [&]() { return LoadWordInBounds(ctx, resource, index); });
}

uint32_t LoadWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadWordPrepared(ctx, inst, mem, resource);
	});
}

uint32_t LoadSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                             const MemoryResourceAccess& resource, uint32_t bits) {
	const auto address = ByteAddress(ctx, inst, mem, resource.byte_offset, resource.stride);
	const auto index = Binary(
	    ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), address,
	    ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	return EmitValueOrZeroIfCondition(
	    ctx.state, EmitStorageElementInBounds(ctx.state, resource, index),
	    [&]() { return LoadSubwordInBounds(ctx, resource, address, index, bits); });
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index) {
	const auto value   = ctx.state.builder.AllocateId();
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), value, pointer,
	                              resource.memory_access);
	return value;
}

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits) {
	uint32_t value;
	if (resource.element_bits < 32u) {
		const auto loaded = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpLoad,
		                              ctx.state.storage_buffers[bits == 8u ? 0 : 1].element_type, loaded,
		                              EmitMemoryElementPointer(ctx.state, resource, index),
		                              resource.memory_access);
		value = Unary(ctx.state, spv::OpUConvert, TypeU32(ctx.state), loaded);
	} else {
		const auto word = LoadWordInBounds(ctx, resource, index);
		const auto byte = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
		                         ConstantU32(ctx.state, 3));
		const auto shift = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state), byte,
		                          ConstantU32(ctx.state, 3));
		value = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state),
		               Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), word, shift),
		               ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu));
	}
	return value;
}

uint32_t LoadSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem,
                     uint32_t bits) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadSubwordPrepared(ctx, inst, mem, resource, bits);
	});
}

Prospero::BufferFormat BufferFormat(const ValueEmitContext& ctx, const IR::MemoryInfo& mem) {
	return mem.typed ? Format::DecodeTBufferFormat(mem.data_format, mem.number_format)
	                 : StorageBufferFormat(ctx.state, mem);
}

IR::MemoryInfo RebaseRawComponent(IR::MemoryInfo mem, uint32_t component) {
	mem.offset += component * 4u;
	mem.data_dwords     = 1u;
	mem.component_index = component;
	return mem;
}

using Format::FormattedSource;
using Format::FormattedSourceKind;

FormattedSource ResolveFormattedSource(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                       const Format::BufferFormatInfo& info,
                                       uint32_t                        output_component) {
	if (mem.typed) {
		return output_component < info.component_count
		           ? FormattedSource {FormattedSourceKind::Memory, output_component}
		           : FormattedSource {};
	}
	const auto selector = GetDstSel(ctx.state.program.info.buffers[mem.resource].descriptor_swizzle,
	                                output_component);
	const auto source = Format::ResolveFormattedSource(info, selector);
	if (source.kind == FormattedSourceKind::Invalid) {
		ExitDescriptorBindingFailure(ctx.state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "buffer descriptor has reserved dst_sel");
	}
	return source;
}

uint32_t FormattedConstant(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                           FormattedSourceKind kind) {
	return ConstantU32(ctx.state, Format::FormattedConstantBits(info, kind));
}

template <typename LoadWordFn, typename LoadSubwordFn>
uint32_t LoadFormattedComponent(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                                FormattedSource source, LoadWordFn&& load_word,
                                LoadSubwordFn&& load_subword) {
	if (source.kind != FormattedSourceKind::Memory) {
		return FormattedConstant(ctx, info, source.kind);
	}
	const auto component = source.component;
	const auto bits      = info.component_bits[component];
	auto raw = info.packed_bitfield || bits == 32u ? load_word(component)
	                                                : load_subword(component, bits);
	if (info.packed_bitfield || (bits < 32u && IsSignedFormatComponent(info.type))) {
		const auto type =
		    IsSignedFormatComponent(info.type) ? TypeI32(ctx.state) : TypeU32(ctx.state);
		const auto source_value =
		    type == TypeI32(ctx.state) ? Unary(ctx.state, spv::OpBitcast, type, raw) : raw;
		const auto extracted = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(IsSignedFormatComponent(info.type) ? spv::OpBitFieldSExtract
		                                                                 : spv::OpBitFieldUExtract,
		                              type, extracted, source_value,
		                              ConstantU32(ctx.state, info.packed_bitfield
		                                                         ? info.component_bit_offset[component]
		                                                         : 0u),
		                              ConstantU32(ctx.state, bits));
		raw = type == TypeI32(ctx.state)
		          ? Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), extracted)
		          : extracted;
	}
	return NormalizeFormatComponent(ctx.state, info, component, raw);
}

void StoreSubwordInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t address, uint32_t index,
                          uint32_t bits, uint32_t data) {
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	if (resource.element_bits < 32u) {
		ctx.state.builder.AddFunction(
		    spv::OpStore, pointer,
		    Unary(ctx.state, spv::OpUConvert,
		          ctx.state.storage_buffers[bits == 8u ? 0 : 1].element_type, data),
		    resource.memory_access);
		return;
	}
	const auto shift   = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
	                                   ConstantU32(ctx.state, 3)),
	                            ConstantU32(ctx.state, 3));
	const auto mask    = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu), shift);
	const auto value   = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), data,
	                                   ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu)),
	                            shift);
	const auto merge   = [&](uint32_t old) {
		return Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state),
		              Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), old,
		                     Unary(ctx.state, spv::OpNot, TypeU32(ctx.state), mask)),
		              value);
	};
	if (mem.kind == IR::ResourceKind::Scratch) {
		const auto old = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), old, pointer);
		ctx.state.builder.AddFunction(spv::OpStore, pointer, merge(old));
	} else {
		AtomicUpdate(ctx.state, pointer, mem.kind, merge);
	}
}

void StoreSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t bits, uint32_t data) {
	if (resource.object_pointer == 0u) return;
	const auto address = ByteAddress(ctx, inst, mem, resource.byte_offset, resource.stride);
	const auto index = Binary(
	    ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), address,
	    ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	// 32-bit views merge subwords in a compare-exchange loop, and out-of-range atomics return
	// undefined values.
	const auto in_bounds = resource.element_bits < 32u
	                           ? EmitStorageElementInBounds(ctx.state, resource, index)
	                           : EmitMemoryElementInBounds(ctx.state, resource, index);
	EmitIfCondition(ctx.state, BufferIndexInBounds(ctx, inst, mem, resource, in_bounds), [&]() {
		StoreSubwordInBounds(ctx, mem, resource, address, index, bits, data);
	});
}

void StoreSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem, uint32_t bits) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		EmitBufferAccess(ctx, inst, mem, 0u, [&](const auto& selected, const auto& resource) {
			StoreSubwordPrepared(ctx, inst, selected, resource, bits, ctx.Arg(inst, inst.NumArgs() - 2));
		});
	});
}

void StoreWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource, uint32_t index,
                       uint32_t data) {
	ctx.state.builder.AddFunction(spv::OpStore,
	                              EmitMemoryElementPointer(ctx.state, resource, index), data,
	                              resource.memory_access);
}

template <typename Fn>
void ForEachLocalFlatAccess(ValueEmitContext& ctx, const IR::Inst& inst, Fn&& emit) {
	auto& state = ctx.state;
	auto mem = ctx.Memory(inst);
	auto high = ctx.Arg(inst, 2);
	const auto low = AddAddressOffset(state, mem, ctx.Arg(inst, 1), high);
	const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), low, ConstantU32(state, 2));
	const auto aligned = Binary(state, spv::OpIEqual, TypeBool(state), ConstantU32(state, 0),
	    Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, 3)));
	for (const auto kind: {IR::ResourceKind::Scratch, IR::ResourceKind::Lds}) {
		mem.kind = kind;
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto aperture = kind == IR::ResourceKind::Scratch ? Decoder::PrivateApertureHigh
		                                                       : Decoder::SharedApertureHigh;
		const auto selected = Binary(state, spv::OpIEqual, TypeBool(state), high, ConstantU32(state, aperture));
		const auto valid = AndCondition(state, aligned,
		    AndCondition(state, selected, EmitMemoryElementInBounds(state, resource, index)));
		emit(resource, index, valid);
	}
}

uint32_t LoadLocalFlat(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		uint32_t result = 0;
		ForEachLocalFlatAccess(ctx, inst, [&](const auto& resource, uint32_t index, uint32_t valid) {
			const auto value = EmitValueOrZeroIfCondition(ctx.state, valid, [&]() {
				return LoadWordInBounds(ctx, resource, index);
			});
			result = result == 0 ? value : Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state), result, value);
		});
		return result;
	});
}

void StoreLocalFlat(ValueEmitContext& ctx, const IR::Inst& inst) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		ForEachLocalFlatAccess(ctx, inst, [&](const auto& resource, uint32_t index, uint32_t valid) {
			EmitIfCondition(ctx.state, valid, [&]() {
				StoreWordInBounds(ctx, resource, index, ctx.Arg(inst, inst.NumArgs() - 2));
			});
		});
	});
}

void StoreWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                       const MemoryResourceAccess& resource, uint32_t data) {
	if (resource.object_pointer == 0u) return;
	const auto index = DwordIndex(ctx, inst, mem, resource.byte_offset, resource.stride);
	const auto in_bounds = BufferIndexInBounds(ctx, inst, mem, resource,
	    EmitStorageElementInBounds(ctx.state, resource, index));
	EmitIfCondition(ctx.state, in_bounds, [&]() {
		StoreWordInBounds(ctx, resource, index, data);
	});
}

void StoreWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		EmitBufferAccess(ctx, inst, mem, 0u, [&](const auto& selected, const auto& resource) {
			StoreWordPrepared(ctx, inst, selected, resource, ctx.Arg(inst, inst.NumArgs() - 2));
		});
	});
}

template <typename Fn>
uint32_t EmitAtomicAccess(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& operation) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		auto index = DwordIndex(ctx, inst, mem, resource.byte_offset, resource.stride);
		if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
			// Keep region-address truncation local to DS atomics.
			index = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), index,
			               ConstantU32(ctx.state, 0x3fffu));
		}
		return EmitValueOrZeroIfCondition(
		    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
			    return operation(EmitMemoryElementPointer(ctx.state, resource, index));
		    });
	});
}

template <typename Fn>
uint32_t EmitAtomicUpdate(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& replacement) {
	const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
	return EmitAtomicAccess(ctx, inst, mem, [&](uint32_t pointer) {
		return AtomicUpdate(ctx.state, pointer, mem.kind, [&](uint32_t old) {
			return replacement(ctx.state, old, value);
		});
	});
}

uint32_t AtomicIncrement(EmitterState& state, uint32_t old, uint32_t limit) {
	// old >= limit ? 0 : old + 1 (unsigned).
	const auto wrap = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), old, limit);
	const auto next = Binary(state, spv::OpIAdd, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, ConstantU32(state, 0), next);
}

uint32_t AtomicDecrement(EmitterState& state, uint32_t old, uint32_t limit) {
	// old == 0 || old > limit ? limit : old - 1 (unsigned).
	const auto zero  = Binary(state, spv::OpIEqual, TypeBool(state), old, ConstantU32(state, 0));
	const auto above = Binary(state, spv::OpUGreaterThan, TypeBool(state), old, limit);
	const auto wrap  = Binary(state, spv::OpLogicalOr, TypeBool(state), zero, above);
	const auto next  = Binary(state, spv::OpISub, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, limit, next);
}

struct PreparedFormattedMemory {
	Format::BufferFormatInfo info;
	MemoryResourceAccess     resource;
	uint32_t                 base = 0;
	uint32_t                 in_bounds = 0;
};

PreparedFormattedMemory PrepareFormattedMemory(ValueEmitContext& ctx, const IR::Inst& inst,
                                               const IR::MemoryInfo& mem,
                                               const MemoryResourceAccess& resource,
                                               const Format::BufferFormatInfo& info) {
	PreparedFormattedMemory plan;
	plan.info      = info;
	plan.resource  = resource;
	const auto start = ByteAddress(ctx, inst, mem, resource.byte_offset, resource.stride);
	// Check the whole, unaligned format span before aligning the actual memory address.
	const auto end = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), start,
	                        ConstantU32(ctx.state, info.byte_size - 1u));
	const auto last_index = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), end,
	                               ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	plan.in_bounds = BufferIndexInBounds(ctx, inst, mem, resource, AndCondition(
	    ctx.state, Binary(ctx.state, spv::OpUGreaterThanEqual, TypeBool(ctx.state), end, start),
	    EmitMemoryElementInBounds(ctx.state, plan.resource, last_index)));
	plan.base = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), start,
	                   ConstantU32(ctx.state, ~(std::min(4u, info.byte_size) - 1u)));
	return plan;
}

std::pair<uint32_t, uint32_t> FormattedComponentAddress(EmitterState& state,
                                                       const PreparedFormattedMemory& plan,
                                                       uint32_t component) {
	// Components advance sequentially from one address; do not reapply buffer swizzling.
	const auto offset = Format::GetFormatComponentByteOffset(plan.info, component);
	const auto address = offset == 0u ? plan.base : Binary(
	    state, spv::OpIAdd, TypeU32(state), plan.base, ConstantU32(state, offset));
	return {address, Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
	                        ConstantU32(state, std::countr_zero(plan.resource.element_bits / 8u)))};
}

uint32_t LoadFormattedInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                               const PreparedFormattedMemory& plan, uint32_t output_component) {
	return LoadFormattedComponent(
	    ctx, plan.info, ResolveFormattedSource(ctx, mem, plan.info, output_component),
	    [&](uint32_t component) {
		    return LoadWordInBounds(ctx, plan.resource,
		                            FormattedComponentAddress(ctx.state, plan, component).second);
	    },
	    [&](uint32_t component, uint32_t bits) {
		    const auto [address, index] = FormattedComponentAddress(ctx.state, plan, component);
		    return LoadSubwordInBounds(ctx, plan.resource, address, index, bits);
	    });
}

uint32_t ConstructU32Composite(EmitterState& state, uint32_t components,
                               const std::array<uint32_t, 4>& values) {
	if (components == 1u) return values[0];
	const auto            result = state.builder.AllocateId();
	std::vector<uint32_t> words {spv::OpCompositeConstruct, TypeU32Composite(state, components),
	                             result};
	words.insert(words.end(), values.begin(), values.begin() + components);
	state.builder.AddFunction(words);
	return result;
}

uint32_t FormattedOutOfBoundsValue(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                   const PreparedFormattedMemory& plan, uint32_t components) {
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; component++) {
		const auto source = ResolveFormattedSource(ctx, mem, plan.info, component);
		values[component] = FormattedConstant(ctx, plan.info, source.kind);
		if (!mem.typed && source.kind == FormattedSourceKind::One &&
		    plan.resource.invalid_format != 0u) {
			values[component] = Select(ctx.state, TypeU32(ctx.state),
			                           plan.resource.invalid_format,
			                           ConstantU32(ctx.state, 0u), values[component]);
		}
	}
	return ConstructU32Composite(ctx.state, components, values);
}

uint32_t LoadFormattedPrepared(ValueEmitContext& ctx, const IR::Inst& inst,
                               const IR::MemoryInfo& mem, const MemoryResourceAccess& resource,
                               const Format::BufferFormatInfo& info, uint32_t components) {
	auto&      state = ctx.state;
	const auto count = mem.data_bits == 16u ? mem.component_count : components;
	const auto plan = resource.object_pointer == 0u ? PreparedFormattedMemory{.info = info}
	                                              : PrepareFormattedMemory(ctx, inst, mem, resource, info);
	const auto out_of_bounds = FormattedOutOfBoundsValue(ctx, mem, plan, count);
	const auto value = resource.object_pointer == 0u ? out_of_bounds : EmitValueOrDefaultIfCondition(
	    state, plan.in_bounds, TypeU32Composite(state, count), out_of_bounds, [&]() {
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < count; component++) {
			    values[component] = LoadFormattedInBounds(ctx, mem, plan, component);
		    }
		    return ConstructU32Composite(state, count, values);
	    });
	if (mem.data_bits != 16u) return value;
	const bool integer = info.type == Format::ComponentType::Uint ||
	                     info.type == Format::ComponentType::Sint;
	const auto scalar = [&](uint32_t component) {
		if (component >= count) return integer ? ConstantU32(state, 0u) : ConstantF32(state, 0);
		auto result = value;
		if (count != 1u) {
			result = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), result, value,
			                          component);
		}
		if (info.type == Format::ComponentType::Uint) {
			return EmitUMin32(state, result, ConstantU32(state, 0xffffu));
		}
		if (info.type == Format::ComponentType::Sint) {
			const auto lower = EmitSMax32(state, result, ConstantU32(state, 0xffff8000u));
			return EmitSMin32(state, lower, ConstantU32(state, 0x7fffu));
		}
		return EmitBitCastF32U32(state, result);
	};
	std::array<uint32_t, 4> packed {};
	for (uint32_t word = 0; word < components; word++) {
		const auto low  = scalar(word * 2u);
		const auto high = scalar(word * 2u + 1u);
		if (integer) {
			packed[word] = Binary(
			    state, spv::OpBitwiseOr, TypeU32(state),
			    Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, 0xffffu)),
			    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), high, ConstantU32(state, 16u)));
		} else {
			const auto pair = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair,
			                          low, high);
			packed[word] = EmitPackHalf2x16(state, pair);
		}
	}
	return ConstructU32Composite(state, components, packed);
}

void StoreFormattedPrepared(ValueEmitContext& ctx, const IR::Inst& inst,
                             const IR::MemoryInfo& mem, const MemoryResourceAccess& resource,
                             const Format::BufferFormatInfo& info, uint32_t data,
                             uint32_t components) {
	if (resource.object_pointer == 0u) return;
	// RDNA2 formatted stores transfer the entire format, zero-filling missing VGPRs.
	const auto count = mem.data_bits == 16u ? mem.component_count : components;
	const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info);
	EmitIfCondition(ctx.state, plan.in_bounds, [&]() {
		auto packed = ConstantU32(ctx.state, 0);
		for (uint32_t component = 0; component < info.component_count; component++) {
			auto value = ConstantU32(ctx.state, 0);
			if (component < count) {
				const auto word = mem.data_bits == 16u ? component / 2u : component;
				value = data;
				if (components != 1u) {
					value = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state),
					                              value, data, word);
				}
				if (mem.data_bits == 16u && (component & 1u) != 0u)
					value = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), value,
					               ConstantU32(ctx.state, 16u));
			}
			value = EncodeFormattedStoreComponent(ctx.state, info, component, value, mem.data_bits);
			const auto bits = info.component_bits[component];
			if (info.packed_bitfield) {
				packed = EmitBitFieldInsert(
				    ctx.state, packed, value,
				    ConstantU32(ctx.state, info.component_bit_offset[component]),
				    ConstantU32(ctx.state, bits));
			} else {
				const auto [address, index] = FormattedComponentAddress(ctx.state, plan, component);
				if (bits == 8u || bits == 16u) {
					StoreSubwordInBounds(ctx, mem, plan.resource, address, index, bits, value);
				} else {
					StoreWordInBounds(ctx, plan.resource, index, value);
				}
			}
		}
		if (info.packed_bitfield) {
			StoreWordInBounds(ctx, plan.resource,
			                  FormattedComponentAddress(ctx.state, plan, 0u).second, packed);
		}
	});
}

uint32_t LoadIndirectScalarBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state  = ctx.state;
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto  word1  = ctx.Arg(handle, 1);
	const auto  stride = EmitBitFieldUExtract(state, word1, ConstantU32(state, 16),
	                                         ConstantU32(state, 14));
	const auto  size   = Binary(state, spv::OpIMul, TypeU64(state),
	                            Unary(state, spv::OpUConvert, TypeU64(state),
	                                  EmitUMax32(state, stride, ConstantU32(state, 1))),
	                            Unary(state, spv::OpUConvert, TypeU64(state), ctx.Arg(handle, 2)));
	const auto  offset = Binary(state, spv::OpIAdd, TypeU64(state),
	                            Unary(state, spv::OpUConvert, TypeU64(state),
	                                  Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                                         ctx.Arg(inst, 1), ConstantU32(state, ~3u))),
	                            ConstantU64(state, ctx.Memory(inst).offset & ~3u));
	const auto  end =
	    Binary(state, spv::OpIAdd, TypeU64(state), offset, ConstantU64(state, sizeof(uint32_t)));
	return EmitValueOrZeroIfCondition(
	    state, Binary(state, spv::OpULessThanEqual, TypeBool(state), end, size), [&]() {
		    const auto base = PackU64(state,
		                              Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                                     ctx.Arg(handle, 0), ConstantU32(state, ~3u)),
		                              Binary(state, spv::OpBitwiseAnd, TypeU32(state), word1,
		                                     ConstantU32(state, 0xffffu)));
		    return LoadBdaDword(ctx, Binary(state, spv::OpIAdd, TypeU64(state), base, offset));
	    });
}

struct IndirectBufferAccess {
	uint32_t address;
	uint32_t offset;
	uint32_t stride;
	uint32_t swizzle;
	uint32_t format;
	uint32_t selector;
	uint32_t mode;
	uint32_t records;
	uint32_t index_in_bounds;
	uint32_t scalar_in_bounds;
	uint32_t raw_records;
	uint32_t raw_index_in_bounds;
};

std::array<uint32_t, 9> IndirectBufferArgs(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	auto offset = ctx.Arg(inst, 2);
	if (ctx.Memory(inst).offset != 0u) {
		const auto immediate = ConstantU32(state, ctx.Memory(inst).offset);
		offset = offset == ConstantU32(state, 0) ? immediate
		    : Binary(state, spv::OpIAdd, TypeU32(state), offset, immediate);
	}
	return {ctx.Arg(handle, 0), ctx.Arg(handle, 1), ctx.Arg(handle, 2), ctx.Arg(handle, 3),
	        ctx.Arg(inst, 1), offset, ctx.Arg(inst, 3), BufferLane(state),
	        ctx.Arg(inst, inst.NumArgs() - 1)};
}

IndirectBufferAccess PrepareIndirectBuffer(EmitterState& state,
                                           const std::array<uint32_t, 8>& args, bool formatted) {
	const auto& [word0, word1, records, word3, input_index, offset, soffset, lane] = args;
	const auto field = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
	};
	const auto nonzero = [&](uint32_t value) {
		return Binary(state, spv::OpINotEqual, TypeBool(state), value, ConstantU32(state, 0));
	};
	const auto stride  = field(word1, 16, 14);
	const auto swizzle = AndCondition(state, nonzero(stride), nonzero(field(word1, 31, 1)));
	const auto index = Binary(
	    state, spv::OpIAdd, TypeU32(state), input_index,
	    Select(state, TypeU32(state), nonzero(field(word3, 23, 1)), lane,
	           ConstantU32(state, 0)));
	const auto address = CalculateBufferAddress(state, index, offset, soffset, 0u, stride, swizzle,
	                                            field(word3, 21, 2));
	const auto base        = PackU64(state, word0, field(word1, 0, 16));
	const auto raw_records = Binary(state, spv::OpISub, TypeU32(state), records, soffset);
	return {
	    .address          = Binary(state, spv::OpIAdd, TypeU64(state), base,
	                               Unary(state, spv::OpUConvert, TypeU64(state), address.byte)),
	    .offset           = address.offset,
	    .stride           = stride,
	    .swizzle          = swizzle,
	    .format           = field(word3, 12, 7),
	    .selector         = formatted ? field(word3, 0, 3) : 0u,
	    .mode             = field(word3, 28, 2),
	    .records          = records,
	    .index_in_bounds  = Binary(state, spv::OpULessThan, TypeBool(state), index, records),
	    .scalar_in_bounds = Binary(state, spv::OpULessThanEqual, TypeBool(state), soffset, records),
	    .raw_records      = raw_records,
	    .raw_index_in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, raw_records),
	};
}

uint32_t IndirectBufferInBounds(EmitterState& state, const IndirectBufferAccess& buffer,
                                uint32_t displacement, uint32_t bytes, bool formatted) {
	const auto offset = displacement == 0u ? buffer.offset : Binary(
	    state, spv::OpIAdd, TypeU32(state), buffer.offset, ConstantU32(state, displacement));
	const auto has_payload = [&](uint32_t size) {
		return AndCondition(
		    state, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), size, bytes),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state), offset,
		           Binary(state, spv::OpISub, TypeU32(state), size, bytes)));
	};
	const auto structured = AndCondition(
	    state, buffer.index_in_bounds,
	    formatted ? has_payload(buffer.stride)
	              : Binary(state, spv::OpULessThan, TypeBool(state), offset, buffer.stride));
	// OOB_SELECT=3 reduces NUM_RECORDS by SOFFSET before its offset/index checks.
	const auto raw = AndCondition(
	    state, buffer.scalar_in_bounds,
	    Select(state, TypeBool(state), buffer.swizzle,
	           AndCondition(state, buffer.raw_index_in_bounds, has_payload(buffer.stride)),
	           has_payload(buffer.raw_records)));
	auto in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 0)),
	    structured, buffer.index_in_bounds);
	in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpULessThan, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	    in_bounds,
	    Select(state, TypeBool(state),
	           Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	           Binary(state, spv::OpINotEqual, TypeBool(state), buffer.records, ConstantU32(state, 0)),
	           raw));
	return in_bounds;
}

uint32_t LoadIndirectBuffer(ValueEmitContext& ctx, const IndirectBufferAccess& buffer,
                            uint32_t exec, uint32_t bits, uint32_t components, bool coherent) {
	auto&      state        = ctx.state;
	const auto valid_format = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
	                                 ConstantU32(state, 0));
	const auto active_valid = AndCondition(state, exec, valid_format);
	const auto base         = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
	                                 ConstantU64(state, ~uint64_t {3}));
	const auto subword = [&](uint32_t word) {
		if (bits == 32u) return word;
		const auto byte = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                         Unary(state, spv::OpUConvert, TypeU32(state), buffer.address),
		                         ConstantU32(state, bits == 8u ? 3u : 2u));
		return EmitBitFieldUExtract(
		    state, word,
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), byte, ConstantU32(state, 3u)),
		    ConstantU32(state, bits));
	};
	if (components == 1u) {
		return EmitValueOrZeroIfCondition(
		    state,
		    AndCondition(state, active_valid,
		                 IndirectBufferInBounds(state, buffer, 0u, ConstantU32(state, bits / 8u),
		                                        false)),
		    [&]() { return subword(LoadBdaDword(ctx, base, coherent)); });
	}

	// Factor component-invariant bounds logic once, then coalesce all components onto at most
	// two 16 KiB BDA page lookups and one single-block PSB load block so multi-component loads
	// avoid quadratic block/fault explosion when drivers inline them.
	const auto bytes           = ConstantU32(state, bits / 8u);
	const auto mode_eq_0       = Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode,
	                                    ConstantU32(state, 0));
	const auto mode_lt_2       = Binary(state, spv::OpULessThan, TypeBool(state), buffer.mode,
	                                    ConstantU32(state, 2));
	const auto mode_eq_2       = Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode,
	                                    ConstantU32(state, 2));
	const auto stride_nonzero  = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.stride,
	                                    ConstantU32(state, 0));
	const auto records_nonzero = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.records,
	                                    ConstantU32(state, 0));
	const auto raw_size =
	    Select(state, TypeU32(state), buffer.swizzle, buffer.stride, buffer.raw_records);
	const auto raw_max_offset = Binary(state, spv::OpISub, TypeU32(state), raw_size, bytes);
	const auto raw_prefix = AndCondition(
	    state,
	    AndCondition(state, buffer.scalar_in_bounds,
	                 Select(state, TypeBool(state), buffer.swizzle, buffer.raw_index_in_bounds,
	                        ConstantBool(state, true))),
	    Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), raw_size, bytes));
	const auto structured_ok = AndCondition(state, buffer.index_in_bounds, stride_nonzero);
	const auto base_ok = Select(
	    state, TypeBool(state), mode_lt_2,
	    Select(state, TypeBool(state), mode_eq_0, structured_ok, buffer.index_in_bounds),
	    Select(state, TypeBool(state), mode_eq_2, records_nonzero, raw_prefix));
	const auto prefix_ok = AndCondition(state, active_valid, base_ok);
	const auto max_offset = Select(
	    state, TypeU32(state), mode_lt_2,
	    Select(state, TypeU32(state), mode_eq_0,
	           Binary(state, spv::OpISub, TypeU32(state), buffer.stride, ConstantU32(state, 1)),
	           ConstantU32(state, UINT32_MAX)),
	    Select(state, TypeU32(state), mode_eq_2, ConstantU32(state, UINT32_MAX), raw_max_offset));

	std::array<uint32_t, 4> comp_in_bounds {};
	uint32_t                any_in_bounds = 0u;
	for (uint32_t component = 0; component < components; ++component) {
		const auto offset = component == 0u
		                        ? buffer.offset
		                        : Binary(state, spv::OpIAdd, TypeU32(state), buffer.offset,
		                                 ConstantU32(state, component * 4u));
		comp_in_bounds[component] = AndCondition(
		    state, prefix_ok,
		    Binary(state, spv::OpULessThanEqual, TypeBool(state), offset, max_offset));
		any_in_bounds = component == 0u
		                    ? comp_in_bounds[0]
		                    : Binary(state, spv::OpLogicalOr, TypeBool(state), any_in_bounds,
		                             comp_in_bounds[component]);
	}

	const auto composite_type = TypeU32Composite(state, components);
	return EmitValueOrDefaultIfCondition(
	    state, any_in_bounds, composite_type, ConstantU32CompositeZero(state, components), [&]() {
		    // Only in-bounds components participate in first_off / last_off so an OOB tail never
		    // faults an unmapped adjacent 16 KiB BDA page.
		    uint32_t first_off = ConstantU32(state, (components - 1u) * 4u);
		    for (uint32_t component = components - 1u; component-- > 0u;) {
			    first_off = Select(state, TypeU32(state), comp_in_bounds[component],
			                       ConstantU32(state, component * 4u), first_off);
		    }
		    uint32_t last_off = ConstantU32(state, 0u);
		    for (uint32_t component = 1u; component < components; ++component) {
			    last_off = Select(state, TypeU32(state), comp_in_bounds[component],
			                      ConstantU32(state, component * 4u), last_off);
		    }
		    const auto base_low      = Unary(state, spv::OpUConvert, TypeU32(state), base);
		    const auto first_low     = Binary(state, spv::OpIAdd, TypeU32(state), base_low, first_off);
		    const auto last_low      = Binary(state, spv::OpIAdd, TypeU32(state), base_low, last_off);
		    const auto first_off_u64 = Unary(state, spv::OpUConvert, TypeU64(state), first_off);
		    const auto last_off_u64  = Unary(state, spv::OpUConvert, TypeU64(state), last_off);
		    const auto first_addr    = Binary(state, spv::OpIAdd, TypeU64(state), base, first_off_u64);
		    const auto last_addr     = Binary(state, spv::OpIAdd, TypeU64(state), base, last_off_u64);
		    constexpr uint32_t kPageBytes = static_cast<uint32_t>(BufferCache::CACHING_PAGESIZE);
		    const auto crosses_page = Binary(
		        state, spv::OpUGreaterThanEqual, TypeBool(state),
		        Binary(state, spv::OpBitwiseXor, TypeU32(state), first_low, last_low),
		        ConstantU32(state, kPageBytes));
		    const auto first = LookupBdaPage(state, first_addr);
		    const auto last  = LookupBdaPage(state, last_addr);
		    const auto first_unbacked =
		        Binary(state, spv::OpIEqual, TypeBool(state), first.entry, ConstantU64(state, 0));
		    const auto last_unbacked = AndCondition(
		        state, crosses_page,
		        Binary(state, spv::OpIEqual, TypeBool(state), last.entry, ConstantU64(state, 0)));
		    const auto any_unbacked =
		        Binary(state, spv::OpLogicalOr, TypeBool(state), first_unbacked, last_unbacked);
		    EmitIfCondition(state, any_unbacked, [&]() {
			    RecordMaskedBdaFault(state, first.page, first_unbacked);
			    RecordMaskedBdaFault(state, last.page, last_unbacked);
		    });
		    const auto any_present =
		        Binary(state, spv::OpLogicalOr, TypeBool(state), first.present, last.present);
		    const auto missing = ConstantU32(
		        state, state.program.info.bvh ? (bits == 32u ? ~0u : ((1u << bits) - 1u)) : 0u);
		    std::array<uint32_t, 4> fallback {};
		    for (uint32_t component = 0; component < components; ++component) {
			    fallback[component] = state.program.info.bvh
			                              ? Select(state, TypeU32(state), comp_in_bounds[component],
			                                       missing, ConstantU32(state, 0))
			                              : ConstantU32(state, 0);
		    }
		    const auto fallback_composite = state.program.info.bvh
		                                        ? ConstructU32Composite(state, components, fallback)
		                                        : ConstantU32CompositeZero(state, components);
		    return EmitValueOrDefaultIfCondition(
		        state, any_present, composite_type, fallback_composite, [&]() {
			        const auto safe_bda =
			            Select(state, TypeU64(state), first.present, first.bda, last.bda);
			        const auto base_from_first =
			            Binary(state, spv::OpISub, TypeU64(state), first.bda, first_off_u64);
			        const auto base_from_last =
			            Binary(state, spv::OpISub, TypeU64(state), last.bda, last_off_u64);
			        std::array<uint32_t, 4> values {};
			        for (uint32_t component = 0; component < components; ++component) {
				        const auto comp_low =
				            component == 0u
				                ? base_low
				                : Binary(state, spv::OpIAdd, TypeU32(state), base_low,
				                         ConstantU32(state, component * 4u));
				        const auto same_first = Binary(
				            state, spv::OpULessThan, TypeBool(state),
				            Binary(state, spv::OpBitwiseXor, TypeU32(state), comp_low, first_low),
				            ConstantU32(state, kPageBytes));
				        const auto present =
				            Select(state, TypeBool(state), same_first, first.present, last.present);
				        const auto ok = AndCondition(state, comp_in_bounds[component], present);
				        const auto comp_base = Select(state, TypeU64(state), same_first,
				                                      base_from_first, base_from_last);
				        const auto comp_ptr =
				            component == 0u
				                ? comp_base
				                : Binary(state, spv::OpIAdd, TypeU64(state), comp_base,
				                         ConstantU64(state, component * 4u));
				        const auto ptr_bda =
				            Select(state, TypeU64(state), ok, comp_ptr, safe_bda);
				        const auto loaded = subword(LoadBdaDwordFromPointer(state, ptr_bda, coherent));
				        values[component] = Select(state, TypeU32(state), ok, loaded,
				                                   fallback[component]);
			        }
			        return ConstructU32Composite(state, components, values);
		        });
	    });
}

uint32_t LoadIndirectFormattedX(ValueEmitContext& ctx, const IndirectBufferAccess& buffer,
                                uint32_t exec, bool d16) {
	auto& state = ctx.state;
	const auto geometry = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, TypeU32Composite(state, 4), geometry,
	                          state.indirect_format_geometry_function, buffer.format);
	const auto field = [&](uint32_t index) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, geometry, index);
		return value;
	};
	const auto bytes = field(0);
	const auto component = Binary(
	    state, spv::OpUMod, TypeU32(state),
	    Binary(state, spv::OpISub, TypeU32(state), buffer.selector, ConstantU32(state, 4)), field(1));
	const auto alignment = Unary(state, spv::OpUConvert, TypeU64(state), field(3));
	const auto base = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
	                         Unary(state, spv::OpNot, TypeU64(state), alignment));
	const auto address = Binary(
	    state, spv::OpIAdd, TypeU64(state), base,
	    Unary(state, spv::OpUConvert, TypeU64(state),
	          Binary(state, spv::OpIMul, TypeU32(state), component, field(2))));
	const auto memory = AndCondition(
	    state, Binary(state, spv::OpINotEqual, TypeBool(state), bytes, ConstantU32(state, 0)),
	    Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), buffer.selector,
	           ConstantU32(state, 4)));
	const auto raw = EmitValueOrZeroIfCondition(
	    state, AndCondition(state, exec,
	                        AndCondition(state, memory, IndirectBufferInBounds(state, buffer, 0u, bytes, true))),
	    [&]() {
		    // Format alignment and component spacing keep every selected component within one DWORD.
		    const auto word = LoadBdaDword(
		        ctx, Binary(state, spv::OpBitwiseAnd, TypeU64(state), address,
		                    ConstantU64(state, ~uint64_t {3})));
		    const auto byte = Binary(
		        state, spv::OpBitwiseAnd, TypeU32(state),
		        Unary(state, spv::OpUConvert, TypeU32(state), address), ConstantU32(state, 3));
		    return Binary(state, spv::OpShiftRightLogical, TypeU32(state), word,
		                  Binary(state, spv::OpShiftLeftLogical, TypeU32(state), byte,
		                         ConstantU32(state, 3)));
	    });
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, TypeU32(state), result,
	                          state.indirect_format_function, buffer.format, buffer.selector,
	                          component, raw);
	uint32_t final_value = result;
	if (d16) {
		final_value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionCall, TypeU32(state), final_value,
		                          state.indirect_format_d16_function, buffer.format, result);
	}
	return Select(state, TypeU32(state), exec, final_value, ConstantU32(state, 0));
}

uint32_t LoadRawBuffer(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                       const MemoryResourceAccess& resource, uint32_t components) {
	auto&      state  = ctx.state;
	const auto scalar = [&] {
		std::array<uint32_t, 4> values {};
		for (uint32_t component = 0; component < components; component++) {
			values[component] =
			    LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource);
		}
		return ConstructU32Composite(state, components, values);
	};
	if (!mem.IsCoherentBufferPairLoad(inst.GetOpcode())) return scalar();

	const auto first  = ByteAddress(ctx, inst, mem, resource.byte_offset, resource.stride);
	const auto second = ByteAddress(ctx, inst, RebaseRawComponent(mem, 1), resource.byte_offset,
	                                resource.stride);
	const auto wide =
	    PrepareStorageBufferResourceAccess(state, mem, state.storage_buffers[3],
	                                       resource.non_uniform ? resource.descriptor_index : 0u);
	const auto index =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), first, ConstantU32(state, 3));
	auto atomic =
	    Binary(state, spv::OpIEqual, TypeBool(state),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), first, ConstantU32(state, 7)),
	           ConstantU32(state, 0));
	atomic = AndCondition(state, atomic,
	                      Binary(state, spv::OpIEqual, TypeBool(state), second,
	                             EmitAddU32(state, first, ConstantU32(state, 4))));
	// The 64-bit alias requires the entire pair; partial and swizzled reads retain DWORD bounds.
	atomic = AndCondition(state, atomic, EmitMemoryElementInBounds(state, wide, index));
	const auto atomic_label = state.builder.AllocateId(), scalar_label = state.builder.AllocateId();
	const auto merge = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, atomic, atomic_label, scalar_label);
	EmitLabel(state, atomic_label);
	const auto pointer = EmitStorageBufferElementPointer(
	    state, wide, index, TypeStorageBufferElementPointer(state, 64));
	const auto snapshot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicLoad, TypeU64(state), snapshot, pointer,
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsAcquireMask |
	                                                 spv::MemorySemanticsUniformMemoryMask));
	const std::array<uint32_t, 4> words {
	    EmitCompositeExtractU64(state, snapshot, IR::Value(0u)),
	    EmitCompositeExtractU64(state, snapshot, IR::Value(1u))};
	const auto pair = ConstructU32Composite(state, 2, words);
	state.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(state, scalar_label);
	const auto fallback    = scalar();
	const auto scalar_exit = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(state, merge);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, TypeU32Composite(state, 2), result, pair, atomic_label,
	                          fallback, scalar_exit);
	return result;
}

uint32_t LoadBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	const auto mem = ctx.Memory(inst);
	if (mem.kind == IR::ResourceKind::IndirectBuffer &&
	    mem.SupportsIndirectBufferLoad(inst.GetOpcode())) {
		const auto function = state.indirect_buffer_load_functions.at(
		    {mem.data_bits, components, uint32_t(mem.coherent), uint32_t(mem.formatted)});
		const auto args = IndirectBufferArgs(ctx, inst);
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionCall, TypeU32Composite(state, components),
		                          result, function, args);
		return result;
	}
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    return EmitBufferAccess(ctx, inst, mem, components, [&](const auto& selected, const auto& resource) {
			    const auto info = Format::GetFormatInfo(
			        selected.formatted ? BufferFormat(ctx, selected) : Prospero::BufferFormat::kInvalid);
			    if (info.type != Format::ComponentType::Unknown) {
				    return LoadFormattedPrepared(ctx, inst, selected, resource, info, components);
			    }
			    if (resource.object_pointer == 0u) return ConstantU32CompositeZero(state, components);
			    return LoadRawBuffer(ctx, inst, selected, resource, components);
		    });
	    });
}

void StoreBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem = ctx.Memory(inst);
		const auto composite = ctx.Arg(inst, inst.NumArgs() - 2);
		EmitBufferAccess(ctx, inst, mem, 0u, [&](const auto& selected, const auto& resource) {
			const auto info = Format::GetFormatInfo(
			    selected.formatted ? BufferFormat(ctx, selected) : Prospero::BufferFormat::kInvalid);
			if (info.type != Format::ComponentType::Unknown) {
				StoreFormattedPrepared(ctx, inst, selected, resource, info, composite, components);
				return;
			}
			for (uint32_t component = 0; component < components; component++) {
				auto data = composite;
				if (components != 1u) {
					data = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), data, composite,
					                          component);
				}
				StoreWordPrepared(ctx, inst, RebaseRawComponent(selected, component), resource, data);
			}
		});
	});
}

uint32_t LoadWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    const auto              mem      = ctx.Memory(inst);
		    const auto              resource = PrepareMemoryResourceAccess(state, mem);
		    const auto              base     = ByteAddress(ctx, inst, mem);
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < components; component++) {
			    const auto address   = component == 0u
			                               ? base
			                               : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                        ConstantU32(state, component * 4u));
			    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state),
			                                  address, ConstantU32(state, 2));
			    values[component]    = EmitValueOrZeroIfCondition(
			        state, EmitMemoryElementInBounds(state, resource, index),
			        [&]() { return LoadWordInBounds(ctx, resource, index); });
		    }
		    return ConstructU32Composite(state, components, values);
	    });
}

void StoreWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem      = ctx.Memory(inst);
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto base     = ByteAddress(ctx, inst, mem);
		for (uint32_t component = 0; component < components; component++) {
			const auto address = component == 0u ? base
			                                     : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                              ConstantU32(state, component * 4u));
			const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
			                              ConstantU32(state, 2));
			EmitIfCondition(state, EmitMemoryElementInBounds(state, resource, index),
			                [&]() {
				                StoreWordInBounds(ctx, resource, index, ctx.Arg(inst, component + 1u));
			                });
		}
	});
}

constexpr uint32_t IndirectBufferFormatCount =
    static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) + 1u;

struct IndirectFormatLayout {
	uint32_t                   byte_size;
	uint32_t                   component_count;
	uint32_t                   component_step;
	uint32_t                   alignment_mask;
	std::array<uint32_t, 4>    component_bits;
	std::array<uint32_t, 4>    component_offset;
	bool                       packed;
};

constexpr IndirectFormatLayout LayoutFromInfo(const Format::BufferFormatInfo& info) {
	IndirectFormatLayout layout {};
	layout.byte_size       = info.byte_size;
	layout.component_count = std::max(1u, info.component_count);
	layout.component_step  = info.packed_bitfield ? 0u : info.component_bits[0] / 8u;
	layout.alignment_mask  = info.byte_size == 0u ? 0u : std::min(4u, info.byte_size) - 1u;
	layout.packed          = info.packed_bitfield;
	for (uint32_t c = 0u; c < 4u; ++c) {
		const uint32_t eff     = (!info.packed_bitfield || c >= info.component_count) ? 0u : c;
		layout.component_bits[c]   = info.component_bits[eff];
		layout.component_offset[c] = info.packed_bitfield ? info.component_bit_offset[eff] : 0u;
	}
	return layout;
}

constexpr uint32_t EncodeWidthCode(uint32_t bits) {
	switch (bits) {
		case 0u: return 0u;
		case 2u: return 1u;
		case 8u: return 2u;
		case 10u: return 3u;
		case 11u: return 4u;
		case 16u: return 5u;
		case 32u: return 6u;
		default: return 7u;
	}
}

struct IndirectFormatTables {
	std::array<uint32_t, (IndirectBufferFormatCount + 3u) / 4u> format_words {};
	std::array<uint32_t, 8>                                     geometry_words {};
	std::array<uint32_t, 16>                                    component_spec_words {};
	uint32_t                                                    layout_count = 0u;
};

constexpr IndirectFormatTables BuildIndirectFormatTables() {
	IndirectFormatTables                     tables {};
	std::array<IndirectFormatLayout, 16>     layouts {};
	const auto                               find_or_add_layout = [&](const IndirectFormatLayout& candidate) {
		for (uint32_t i = 0u; i < tables.layout_count; ++i) {
			const auto& existing = layouts[i];
			if (existing.byte_size == candidate.byte_size &&
			    existing.component_count == candidate.component_count &&
			    existing.component_step == candidate.component_step &&
			    existing.alignment_mask == candidate.alignment_mask &&
			    existing.packed == candidate.packed &&
			    existing.component_bits == candidate.component_bits &&
			    existing.component_offset == candidate.component_offset) {
				return i;
			}
		}
		const uint32_t idx = tables.layout_count++;
		layouts[idx]       = candidate;
		return idx;
	};

	for (uint32_t f = 0u; f < IndirectBufferFormatCount; ++f) {
		const auto info      = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(f));
		const auto type_code = static_cast<uint32_t>(info.type);
		const auto layout_id = find_or_add_layout(LayoutFromInfo(info));
		const auto byte_val  = ((layout_id & 0x1fu) << 3u) | (type_code & 0x7u);
		tables.format_words[f >> 2u] |= byte_val << ((f & 3u) * 8u);
	}
	for (uint32_t i = 0u; i < tables.layout_count; ++i) {
		const auto& l = layouts[i];
		const uint32_t geom16 = (l.byte_size & 0x1fu) | ((l.component_count & 0x7u) << 5u) |
		                        ((l.component_step & 0x7u) << 8u) |
		                        ((l.alignment_mask & 0x7u) << 11u);
		tables.geometry_words[i >> 1u] |= geom16 << ((i & 1u) * 16u);
		uint32_t spec = 0u;
		for (uint32_t c = 0u; c < 4u; ++c) {
			const uint32_t code = EncodeWidthCode(l.component_bits[c]);
			const uint32_t byte = (l.component_offset[c] & 0x1fu) | ((code & 0x7u) << 5u);
			spec |= byte << (c * 8u);
		}
		tables.component_spec_words[i] = spec;
	}
	return tables;
}

constexpr IndirectFormatTables kIndirectFormatTables = BuildIndirectFormatTables();

constexpr bool VerifyIndirectFormatTables() {
	if (kIndirectFormatTables.layout_count > 16u) return false;
	for (uint32_t f = 0u; f < IndirectBufferFormatCount; ++f) {
		const auto info      = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(f));
		const auto expected  = LayoutFromInfo(info);
		const auto byte_val  = (kIndirectFormatTables.format_words[f >> 2u] >> ((f & 3u) * 8u)) & 0xffu;
		const auto type_code = byte_val & 0x7u;
		const auto layout_id = (byte_val >> 3u) & 0x1fu;
		if (type_code != static_cast<uint32_t>(info.type) || layout_id >= kIndirectFormatTables.layout_count) {
			return false;
		}
		const auto geom16 = (kIndirectFormatTables.geometry_words[layout_id >> 1u] >> ((layout_id & 1u) * 16u)) &
		                    0xffffu;
		if ((geom16 & 0x1fu) != expected.byte_size ||
		    ((geom16 >> 5u) & 0x7u) != expected.component_count ||
		    ((geom16 >> 8u) & 0x7u) != expected.component_step ||
		    ((geom16 >> 11u) & 0x7u) != expected.alignment_mask) {
			return false;
		}
		const auto spec = kIndirectFormatTables.component_spec_words[layout_id];
		for (uint32_t c = 0u; c < 4u; ++c) {
			const auto cb = (spec >> (c * 8u)) & 0xffu;
			if ((cb & 0x1fu) != expected.component_offset[c] ||
			    ((cb >> 5u) & 0x7u) != EncodeWidthCode(expected.component_bits[c])) {
				return false;
			}
		}
	}
	return true;
}

static_assert(VerifyIndirectFormatTables(), "indirect buffer format lookup table mismatch");

uint32_t EmitConstantU32TableLookup(EmitterState& state, uint32_t index,
                                    std::span<const uint32_t> words, uint32_t base = 0u) {
	if (words.size() == 1u) {
		return ConstantU32(state, words.front());
	}
	const size_t   mid  = words.size() / 2u;
	const uint32_t low  = EmitConstantU32TableLookup(state, index, words.first(mid), base);
	const uint32_t high = EmitConstantU32TableLookup(state, index, words.subspan(mid),
	                                                 base + static_cast<uint32_t>(mid));
	const uint32_t pick = Binary(state, spv::OpULessThan, TypeBool(state), index,
	                             ConstantU32(state, base + static_cast<uint32_t>(mid)));
	return Select(state, TypeU32(state), pick, low, high);
}

uint32_t EmitPackedFormatByte(EmitterState& state, uint32_t format) {
	const auto u           = TypeU32(state);
	const auto safe_format = Select(
	    state, u, Binary(state, spv::OpULessThan, TypeBool(state), format, ConstantU32(state, IndirectBufferFormatCount)),
	    format, ConstantU32(state, 0u));
	const auto word = EmitConstantU32TableLookup(
	    state, Binary(state, spv::OpShiftRightLogical, u, safe_format, ConstantU32(state, 2u)),
	    kIndirectFormatTables.format_words);
	const auto bit_shift = Binary(
	    state, spv::OpShiftLeftLogical, u,
	    Binary(state, spv::OpBitwiseAnd, u, safe_format, ConstantU32(state, 3u)),
	    ConstantU32(state, 3u));
	return EmitBitFieldUExtract(state, word, bit_shift, ConstantU32(state, 8u));
}

void DefineIndirectBufferFormats(ValueEmitContext& ctx) {
	auto&      state = ctx.state;
	const auto u     = TypeU32(state);
	const auto i     = TypeI32(state);
	const auto f     = TypeF32(state);
	const auto b     = TypeBool(state);
	{
		const auto result_type   = TypeU32Composite(state, 4);
		const auto function_type = state.builder.Type(spv::OpTypeFunction, result_type, u);
		state.indirect_format_geometry_function = state.builder.AllocateId();
		const auto format = state.builder.AllocateId();
		state.builder.AddName(state.indirect_format_geometry_function,
		                      "indirect_buffer_format_geometry");
		state.builder.AddFunction(spv::OpFunction, result_type,
		                          state.indirect_format_geometry_function,
		                          spv::FunctionControlMaskNone, function_type);
		state.builder.AddFunction(spv::OpFunctionParameter, u, format);
		EmitLabel(state, state.builder.AllocateId());
		const auto format_byte = EmitPackedFormatByte(state, format);
		const auto layout_id   = EmitBitFieldUExtract(state, format_byte, ConstantU32(state, 3u),
		                                              ConstantU32(state, 5u));
		const auto geom_word   = EmitConstantU32TableLookup(
		    state, Binary(state, spv::OpShiftRightLogical, u, layout_id, ConstantU32(state, 1u)),
		    kIndirectFormatTables.geometry_words);
		const auto geom_shift = Binary(
		    state, spv::OpShiftLeftLogical, u,
		    Binary(state, spv::OpBitwiseAnd, u, layout_id, ConstantU32(state, 1u)),
		    ConstantU32(state, 4u));
		const auto geom16 = EmitBitFieldUExtract(state, geom_word, geom_shift, ConstantU32(state, 16u));
		const std::array values {
		    EmitBitFieldUExtract(state, geom16, ConstantU32(state, 0u), ConstantU32(state, 5u)),
		    EmitBitFieldUExtract(state, geom16, ConstantU32(state, 5u), ConstantU32(state, 3u)),
		    EmitBitFieldUExtract(state, geom16, ConstantU32(state, 8u), ConstantU32(state, 3u)),
		    EmitBitFieldUExtract(state, geom16, ConstantU32(state, 11u), ConstantU32(state, 3u)),
		};
		state.builder.AddFunction(spv::OpReturnValue, ConstructU32Composite(state, 4u, values));
		state.builder.AddFunction(spv::OpFunctionEnd);
	}
	const auto function_type = state.builder.Type(spv::OpTypeFunction, u, u, u, u, u);
	state.indirect_format_function = state.builder.AllocateId();
	const auto format    = state.builder.AllocateId();
	const auto selector  = state.builder.AllocateId();
	const auto component = state.builder.AllocateId();
	const auto raw       = state.builder.AllocateId();
	state.builder.AddName(state.indirect_format_function, "convert_indirect_buffer_format");
	state.builder.AddFunction(spv::OpFunction, u, state.indirect_format_function,
	                          spv::FunctionControlMaskNone, function_type);
	state.builder.AddFunction(spv::OpFunctionParameter, u, format);
	state.builder.AddFunction(spv::OpFunctionParameter, u, selector);
	state.builder.AddFunction(spv::OpFunctionParameter, u, component);
	state.builder.AddFunction(spv::OpFunctionParameter, u, raw);
	EmitLabel(state, state.builder.AllocateId());

	const auto eq_u32 = [&](uint32_t lhs, uint32_t rhs) {
		return Binary(state, spv::OpIEqual, b, lhs, ConstantU32(state, rhs));
	};
	const auto or_b = [&](uint32_t lhs, uint32_t rhs) {
		return Binary(state, spv::OpLogicalOr, b, lhs, rhs);
	};
	const auto format_byte = EmitPackedFormatByte(state, format);
	const auto type_code   = Binary(state, spv::OpBitwiseAnd, u, format_byte, ConstantU32(state, 7u));
	const auto layout_id   = EmitBitFieldUExtract(state, format_byte, ConstantU32(state, 3u),
	                                              ConstantU32(state, 5u));
	const auto comp_spec   = EmitConstantU32TableLookup(
	    state, layout_id, std::span(kIndirectFormatTables.component_spec_words).first(kIndirectFormatTables.layout_count));
	const auto safe_comp   = Select(
	    state, u, Binary(state, spv::OpULessThan, b, component, ConstantU32(state, 4u)),
	    component, ConstantU32(state, 0u));
	const auto spec_byte   = EmitBitFieldUExtract(
	    state, comp_spec,
	    Binary(state, spv::OpShiftLeftLogical, u, safe_comp, ConstantU32(state, 3u)),
	    ConstantU32(state, 8u));
	const auto bit_offset = Binary(state, spv::OpBitwiseAnd, u, spec_byte, ConstantU32(state, 0x1fu));
	const auto width_code = EmitBitFieldUExtract(state, spec_byte, ConstantU32(state, 5u),
	                                             ConstantU32(state, 3u));

	const std::array width_table {0u, 2u, 8u, 10u, 11u, 16u, 32u, 0u};
	const auto       bits = EmitConstantU32TableLookup(state, width_code, width_table);
	const auto       subword = Binary(
	    state, spv::OpLogicalAnd, b,
	    Binary(state, spv::OpUGreaterThanEqual, b, width_code, ConstantU32(state, 1u)),
	    Binary(state, spv::OpULessThanEqual, b, width_code, ConstantU32(state, 5u)));
	const auto safe_bits   = Select(state, u, subword, bits, ConstantU32(state, 1u));
	const auto safe_offset = Select(state, u, subword, bit_offset, ConstantU32(state, 0u));

	const auto is_signed = or_b(
	    eq_u32(type_code, static_cast<uint32_t>(Format::ComponentType::Sint)),
	    or_b(eq_u32(type_code, static_cast<uint32_t>(Format::ComponentType::Snorm)),
	         eq_u32(type_code, static_cast<uint32_t>(Format::ComponentType::Sscaled))));
	const auto u_ext = EmitBitFieldUExtract(state, raw, safe_offset, safe_bits);
	const auto s_ext_i = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitFieldSExtract, i, s_ext_i,
	                          Unary(state, spv::OpBitcast, i, raw), safe_offset, safe_bits);
	const auto s_ext = Unary(state, spv::OpBitcast, u, s_ext_i);
	const auto extracted = Select(state, u, eq_u32(width_code, 6u), raw,
	                              Select(state, u, is_signed, s_ext, u_ext));

	const auto f_from_u = Unary(state, spv::OpConvertUToF, f, extracted);
	const auto f_from_s = Unary(state, spv::OpConvertSToF, f, Unary(state, spv::OpBitcast, i, extracted));
	const auto f_int    = Select(state, f, is_signed, f_from_s, f_from_u);

	const std::array unorm_inv_words {
	    std::bit_cast<uint32_t>(1.0f),           std::bit_cast<uint32_t>(1.0f / 3.0f),
	    std::bit_cast<uint32_t>(1.0f / 255.0f),  std::bit_cast<uint32_t>(1.0f / 1023.0f),
	    std::bit_cast<uint32_t>(1.0f / 2047.0f), std::bit_cast<uint32_t>(1.0f / 65535.0f),
	    std::bit_cast<uint32_t>(1.0f),           std::bit_cast<uint32_t>(1.0f)};
	const std::array snorm_inv_words {
	    std::bit_cast<uint32_t>(1.0f),           std::bit_cast<uint32_t>(1.0f),
	    std::bit_cast<uint32_t>(1.0f / 127.0f),  std::bit_cast<uint32_t>(1.0f / 511.0f),
	    std::bit_cast<uint32_t>(1.0f / 1023.0f), std::bit_cast<uint32_t>(1.0f / 32767.0f),
	    std::bit_cast<uint32_t>(1.0f),           std::bit_cast<uint32_t>(1.0f)};
	const auto is_snorm  = eq_u32(type_code, static_cast<uint32_t>(Format::ComponentType::Snorm));
	const auto unorm_inv = Unary(state, spv::OpBitcast, f,
	                             EmitConstantU32TableLookup(state, width_code, unorm_inv_words));
	const auto snorm_inv = Unary(state, spv::OpBitcast, f,
	                             EmitConstantU32TableLookup(state, width_code, snorm_inv_words));
	const auto norm_f    = Binary(state, spv::OpFMul, f, f_int,
	                              Select(state, f, is_snorm, snorm_inv, unorm_inv));
	const auto clamped_f   = EmitGlsl<GLSLstd450FMax, IR::Type::F32>(state, norm_f,
	                                                                 ConstantF32Value(state, -1.0f));
	const auto norm_bits   = Unary(state, spv::OpBitcast, u, Select(state, f, is_snorm, clamped_f, norm_f));
	const auto scaled_bits = Unary(state, spv::OpBitcast, u, f_int);

	const auto f16_bits  = EmitBitcastF32ToU32(state, EmitF16BitsToF32(state, extracted));
	const auto uf11_bits = NormalizeFormatComponent(
	    ctx.state, Format::GetFormatInfo(Prospero::BufferFormat::k11_11_10Float), 0u, extracted);
	const auto uf10_bits = NormalizeFormatComponent(
	    ctx.state, Format::GetFormatInfo(Prospero::BufferFormat::k11_11_10Float), 2u, extracted);
	const auto float_bits = Select(
	    state, u, eq_u32(width_code, 6u), raw,
	    Select(state, u, eq_u32(width_code, 5u), f16_bits,
	           Select(state, u, eq_u32(width_code, 4u), uf11_bits, uf10_bits)));

	const auto is_int    = or_b(eq_u32(type_code, static_cast<uint32_t>(Format::ComponentType::Uint)),
	                            eq_u32(type_code, static_cast<uint32_t>(Format::ComponentType::Sint)));
	const auto is_norm   = or_b(eq_u32(type_code, static_cast<uint32_t>(Format::ComponentType::Unorm)),
	                            is_snorm);
	const auto is_scaled = or_b(eq_u32(type_code, static_cast<uint32_t>(Format::ComponentType::Uscaled)),
	                            eq_u32(type_code, static_cast<uint32_t>(Format::ComponentType::Sscaled)));
	const auto mem_value = Select(
	    state, u, is_int, extracted,
	    Select(state, u, is_norm, norm_bits,
	           Select(state, u, is_scaled, scaled_bits, float_bits)));

	const auto one_bits    = Select(state, u, is_int, ConstantU32(state, 1u), ConstantU32(state, 0x3f800000u));
	const auto const_value = Select(state, u, eq_u32(selector, 1u), one_bits, ConstantU32(state, 0u));
	const auto chosen      = Select(
	    state, u, Binary(state, spv::OpUGreaterThanEqual, b, selector, ConstantU32(state, 4u)),
	    mem_value, const_value);
	const auto result = Select(
	    state, u, eq_u32(type_code, static_cast<uint32_t>(Format::ComponentType::Unknown)),
	    ConstantU32(state, 0u), chosen);
	state.builder.AddFunction(spv::OpReturnValue, result);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

void DefineIndirectBufferD16(ValueEmitContext& ctx) {
	auto&      state         = ctx.state;
	const auto u             = TypeU32(state);
	const auto b             = TypeBool(state);
	const auto function_type = state.builder.Type(spv::OpTypeFunction, u, u, u);
	state.indirect_format_d16_function = state.builder.AllocateId();
	const auto format = state.builder.AllocateId();
	const auto value  = state.builder.AllocateId();
	state.builder.AddName(state.indirect_format_d16_function, "pack_indirect_buffer_d16");
	state.builder.AddFunction(spv::OpFunction, u, state.indirect_format_d16_function,
	                          spv::FunctionControlMaskNone, function_type);
	state.builder.AddFunction(spv::OpFunctionParameter, u, format);
	state.builder.AddFunction(spv::OpFunctionParameter, u, value);
	EmitLabel(state, state.builder.AllocateId());

	const auto format_byte = EmitPackedFormatByte(state, format);
	const auto type_code   = Binary(state, spv::OpBitwiseAnd, u, format_byte, ConstantU32(state, 7u));
	const auto uint_res    = EmitUMin32(state, value, ConstantU32(state, 0xffffu));
	const auto clamped     = EmitSMin32(state, EmitSMax32(state, value, ConstantU32(state, 0xffff8000u)),
	                                    ConstantU32(state, 0x7fffu));
	const auto sint_res    = Binary(state, spv::OpBitwiseAnd, u, clamped, ConstantU32(state, 0xffffu));
	const auto pair        = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair,
	                          EmitBitCastF32U32(state, value), ConstantF32(state, 0));
	const auto float_res = EmitPackHalf2x16(state, pair);
	const auto result    = Select(
	    state, u,
	    Binary(state, spv::OpIEqual, b, type_code, ConstantU32(state, static_cast<uint32_t>(Format::ComponentType::Uint))),
	    uint_res,
	    Select(state, u,
	           Binary(state, spv::OpIEqual, b, type_code, ConstantU32(state, static_cast<uint32_t>(Format::ComponentType::Sint))),
	           sint_res, float_res));
	state.builder.AddFunction(spv::OpReturnValue, result);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

} // namespace

uint32_t GetBdaPointer(EmitterState& state, uint32_t address) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, TypeU64(state), result,
	                          state.bda_pointer_function, address);
	return result;
}

void DefineIndirectBufferLoads(ValueEmitContext& ctx) {
	auto& state = ctx.state;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			const auto components = IR::BufferComponentCount(inst.GetOpcode());
			if (components == 0u || inst.GetOpcode() == IR::ValueOpcode::ReadConstBuffer) continue;
			const auto& memory = ctx.Memory(inst);
			if (memory.kind != IR::ResourceKind::IndirectBuffer ||
			    !memory.SupportsIndirectBufferLoad(inst.GetOpcode())) continue;
			auto [entry, inserted] = state.indirect_buffer_load_functions.try_emplace(
			    std::array {memory.data_bits, components, uint32_t(memory.coherent),
			                uint32_t(memory.formatted)}, 0u);
			if (!inserted) continue;
			if (memory.formatted && state.indirect_format_function == 0u) {
				DefineIndirectBufferFormats(ctx);
			}
			if (memory.formatted && memory.data_bits == 16u &&
			    state.indirect_format_d16_function == 0u) {
				DefineIndirectBufferD16(ctx);
			}
			const auto function = entry->second = state.builder.AllocateId();
			const auto type = TypeU32(state);
			const auto bool_type = TypeBool(state);
			const auto result_type = TypeU32Composite(state, components);
			const auto signature = state.builder.Type(
			    spv::OpTypeFunction, result_type, type, type, type, type, type, type, type, type,
			    bool_type);
			const auto name = "load_indirect_buffer_" + std::to_string(memory.data_bits) + "x" +
			                  std::to_string(components) + (memory.coherent ? "_coherent" : "") +
			                  (memory.formatted ? "_formatted" : "");
			state.builder.AddName(function, name.c_str());
			state.builder.AddFunction(spv::OpFunction, result_type, function,
			                          spv::FunctionControlMaskNone, signature);
			std::array<uint32_t, 8> args;
			for (auto& arg: args) {
				arg = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFunctionParameter, type, arg);
			}
			const auto exec = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpFunctionParameter, bool_type, exec);
			EmitLabel(state, state.builder.AllocateId());
			const auto buffer = PrepareIndirectBuffer(state, args, memory.formatted);
			const auto result = memory.formatted
			    ? LoadIndirectFormattedX(ctx, buffer, exec, memory.data_bits == 16u)
			    : LoadIndirectBuffer(ctx, buffer, exec, memory.data_bits, components, memory.coherent);
			state.builder.AddFunction(spv::OpReturnValue, result);
			state.builder.AddFunction(spv::OpFunctionEnd);
		}
	}
}

void DefineGetBdaPointer(EmitterState& state) {
	if (!state.program.info.uses_dma) {
		return;
	}
	const auto type            = TypeU64(state);
	const auto function_type   = state.builder.Type(spv::OpTypeFunction, type, type);
	state.bda_pointer_function = state.builder.AllocateId();
	const auto address         = state.builder.AllocateId();
	const auto entry_label     = state.builder.AllocateId();
	state.builder.AddName(state.bda_pointer_function, "get_bda_pointer");
	state.builder.AddFunction(spv::OpFunction, type, state.bda_pointer_function,
	                          spv::FunctionControlMaskNone, function_type);
	state.builder.AddFunction(spv::OpFunctionParameter, type, address);
	EmitLabel(state, entry_label);

	const auto extended = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), address,
	                             ConstantU64(state, LibKernel::Memory::kExtendedMemoryBase));
	const auto packed   = Select(
	    state, type, extended,
	    Binary(state, spv::OpISub, type, address,
	           ConstantU64(state, LibKernel::Memory::kExtendedMemoryBase - LOWER_ADDRESS_SIZE)),
	    address);
	const auto page64        = Binary(state, spv::OpShiftRightLogical, type, packed,
	                                  ConstantU32(state, BufferCache::CACHING_PAGEBITS));
	const auto page          = Unary(state, spv::OpUConvert, TypeU32(state), page64);
	const auto entry_pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state, 64),
	                          entry_pointer, state.bda_pagetable_variable, ConstantU32(state, 0),
	                          page);
	const auto entry = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, type, entry, entry_pointer);
	const auto unbacked =
	    Binary(state, spv::OpIEqual, TypeBool(state), entry, ConstantU64(state, 0));
	const auto fault_label = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, unbacked, fault_label, merge_label);

	EmitLabel(state, fault_label);
	RecordBdaFault(state, page);
	state.builder.AddFunction(spv::OpReturnValue, ConstantU64(state, 0));

	EmitLabel(state, merge_label);
	const auto offset = Binary(state, spv::OpBitwiseAnd, type, address,
	                           ConstantU64(state, BufferCache::CACHING_PAGESIZE - 1));
	state.builder.AddFunction(spv::OpReturnValue, Binary(state, spv::OpIAdd, type, entry, offset));
	state.builder.AddFunction(spv::OpFunctionEnd);
}

uint32_t EmitAtomic32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem = ctx.Memory(inst);
	return EmitAtomicAccess(ctx, inst, mem, [&](uint32_t pointer) {
		const auto scope =
		    mem.kind == IR::ResourceKind::Lds ? spv::ScopeWorkgroup : spv::ScopeDevice;
		const auto old = EmitAtomicOperation(ctx, inst, pointer, scope);
		EmitAtomicMemoryBarrier(ctx.state, mem.kind);
		return old;
	});
}

uint32_t EmitBufferAtomic64(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem   = ctx.Memory(inst);
	auto&       state = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU64(state), ConstantU64(state, 0), [&]() {
		    const auto resource =
		        PrepareStorageBufferResourceAccess(state, mem, state.storage_buffers[3]);
		    const auto byte_address =
		        ByteAddress(ctx, inst, mem, resource.byte_offset, resource.stride);
		    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), byte_address,
		                              ConstantU32(state, 3u));
		    return EmitValueOrDefaultIfCondition(
		        state, EmitMemoryElementInBounds(state, resource, index), TypeU64(state),
		        ConstantU64(state, 0), [&]() {
			        const auto pointer = EmitStorageBufferElementPointer(
			            state, resource, index, TypeStorageBufferElementPointer(state, 64));
			        const auto old = EmitAtomicOperation(ctx, inst, pointer, spv::ScopeDevice);
			        EmitAtomicMemoryBarrier(state, mem.kind);
			        return old;
		        });
	    });
}

void EmitSharedAtomic64(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	const auto& mem = ctx.Memory(inst);
	EnsureLdsStorage(state);
	const auto address = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                            ByteAddress(ctx, inst, mem), ConstantU32(state, 0xfff8u));
	const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
	                          ConstantU32(state, 3u));
	const auto in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index,
	                              ConstantU32(state, LdsDwordCount(state) / 2u));
	const auto active = EmitLogicalAndBool(state, ctx.Arg(inst, inst.NumArgs() - 1), in_bounds);
	const auto emit = [&]() {
		auto native_index = index;
		auto semantics = spv::MemorySemanticsWorkgroupMemoryMask;
		if (state.lds_storage_class == spv::StorageClassStorageBuffer) {
			native_index = EmitAddU32(state, index, EmitBinaryU32(
			    state, spv::OpShiftRightLogical, state.lds_base_dwords, ConstantU32(state, 1)));
			semantics = spv::MemorySemanticsUniformMemoryMask;
		}
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, state.lds_storage_class, TypeU64(state)),
		    pointer, state.lds_u64_variable, ConstantU32(state, 0), native_index);
		const auto old = state.builder.AllocateId();
		const auto opcode = SpirvAtomicOpcode(inst.GetOpcode());
		const auto scope = ConstantU32(state, spv::ScopeWorkgroup);
		const auto acquire_release =
		    ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | semantics);
		if (opcode == spv::OpAtomicCompareExchange) {
			// RDNA2 DS compares DATA and stores DATA2, opposite to BUFFER atomics.
			state.builder.AddFunction(
			    opcode, TypeU64(state), old, pointer, scope, acquire_release,
			    ConstantU32(state, spv::MemorySemanticsAcquireMask | semantics),
			    ctx.Arg(inst, 2), ctx.Arg(inst, 1));
		} else {
			state.builder.AddFunction(opcode, TypeU64(state), old, pointer, scope,
			                          acquire_release, ctx.Arg(inst, 1));
		}
		return old;
	};
	if (inst.HasUses()) {
		ctx.Define(inst, EmitValueOrDefaultIfCondition(
		    state, active, TypeU64(state), ConstantU64(state, 0), emit));
	} else {
		EmitIfCondition(state, active, emit);
	}
}

uint32_t EmitBufferFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem       = ctx.Memory(inst);
	const bool  max_value = inst.GetOpcode() == IR::ValueOpcode::BufferAtomicFMax32;
	return EmitAtomicUpdate(ctx, inst, mem,
	                        [max_value](EmitterState& state, uint32_t old, uint32_t value) {
		                        return EmitFloatAtomicReplacement(state, old, value, max_value);
	                        });
}

void EmitSharedFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem       = ctx.Memory(inst);
	const bool  max_value = inst.GetOpcode() == IR::ValueOpcode::SharedAtomicFMax32;
	EmitAtomicUpdate(ctx, inst, mem,
	                 [max_value](EmitterState& state, uint32_t old, uint32_t value) {
		                 return EmitDsFloatAtomicReplacement(state, old, value, max_value);
	                 });
}

uint32_t EmitAppendConsume(ValueEmitContext& ctx, const IR::Inst& inst) {
	const bool append = inst.GetOpcode() == IR::ValueOpcode::DataAppend;
	auto&      state  = ctx.state;
	if (ctx.half == 1) {
		return ctx.other_half->Def(IR::Value(const_cast<IR::Inst*>(&inst)));
	}
	const auto mem           = ctx.Memory(inst);
	const auto offset        = mem.offset & 0xfffcu;
	auto       address       = ConstantU32(state, offset);
	uint32_t   region_bounds = 0;
	if (mem.kind == IR::ResourceKind::Gds) {
		const auto m0 = ctx.Arg(inst, 0);
		const auto base =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), m0, ConstantU32(state, 16));
		const auto size =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), m0, ConstantU32(state, 0xffffu));
		address = Binary(state, spv::OpIAdd, TypeU32(state), base, address);
		// The entire M0 region must fit the PS5's 48 KiB GDS partition.
		region_bounds = AndCondition(
		    state, Binary(state, spv::OpULessThan, TypeBool(state),
		                  ConstantU32(state, offset + 3u), size),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state),
		           Binary(state, spv::OpIAdd, TypeU32(state), base, size),
		           ConstantU32(state, 0xc000u)));
	}
	const auto index =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), address, ConstantU32(state, 2));
	const auto access = PrepareMemoryResourceAccess(state, mem);
	// Elect and count from the guest EXEC words, which V_MBCNT also ranks lanes by.
	const auto low   = ctx.Arg(inst, 1);
	const auto high  = ctx.Arg(inst, 2);
	const auto count = Binary(state, spv::OpIAdd, TypeU32(state),
	                          Unary(state, spv::OpBitCount, TypeU32(state), low),
	                          Unary(state, spv::OpBitCount, TypeU32(state), high));
	const auto mask  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), mask, low, high,
	                          ConstantU32(state, 0), ConstantU32(state, 0));
	const auto first = ctx.FirstLane(mask);
	const auto source_lane =
	    state.lane_count == 2
	        ? Binary(state, spv::OpBitwiseAnd, TypeU32(state), first, ConstantU32(state, 31))
	        : first;
	const auto is_first       = Binary(state, spv::OpIEqual, TypeBool(state),
	                                   EmitSubgroupLocalInvocationId(state), source_lane);
	auto bounds = EmitMemoryElementInBounds(state, access, index);
	if (mem.kind == IR::ResourceKind::Gds) {
		bounds = AndCondition(state, bounds, region_bounds);
	}
	const auto condition = AndCondition(
	    state, is_first,
	    AndCondition(state,
	                 Binary(state, spv::OpINotEqual, TypeBool(state), count, ConstantU32(state, 0)),
	                 bounds));
	const auto atomic = EmitValueOrZeroIfCondition(state, condition, [&]() {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(append ? spv::OpAtomicIAdd : spv::OpAtomicISub, TypeU32(state),
		                          value, EmitMemoryElementPointer(state, access, index),
		                          ConstantU32(state, mem.kind == IR::ResourceKind::Gds
		                                                 ? spv::ScopeDevice
		                                                 : spv::ScopeWorkgroup),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone), count);
		return value;
	});
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), atomic, source_lane);
	return result;
}

uint32_t EmitReadConst(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (state.flattened_srt_variable == 0) {
		ctx.Fail(inst, "requires the flattened SRT descriptor");
	}
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.flattened_srt_variable, ConstantU32(state, 0),
	                          ctx.Arg(inst, 1));
	return EmitNative<spv::OpLoad, IR::Type::U32>(state, pointer);
}

void EmitReadConstBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto mem = ctx.Memory(inst);
	if (mem.planning_only) return;
	if (mem.kind == IR::ResourceKind::IndirectBuffer) {
		ctx.Define(inst, LoadIndirectScalarBuffer(ctx, inst));
		return;
	}
	auto& state        = ctx.state;
	mem.kind           = IR::ResourceKind::ScalarBuffer;
	const auto offset = inst.Arg(1).Resolve();
	auto       index  = offset.IsImmediate()
	                        ? ConstantU32(state, (offset.U32() >> 2u) + (mem.offset >> 2u))
	                        : Binary(state, spv::OpShiftRightLogical, TypeU32(state),
	                                 ctx.Arg(inst, 1), ConstantU32(state, 2));
	if (!offset.IsImmediate() && mem.offset >= 4u) {
		index = Binary(state, spv::OpIAdd, TypeU32(state), index, ConstantU32(state, mem.offset >> 2u));
	}
	const auto access    = PrepareMemoryResourceAccess(state, mem);
	// Scalar buffer addressing aligns the base and each offset independently.
	auto element = Binary(
	    state, spv::OpIAdd, TypeU32(state), index,
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), access.byte_offset, ConstantU32(state, 2)));
	// The device computes byte offsets in 32 bits, so elements from 2^30 on would wrap back into
	// the range; the host prefix adds at most 63.
	constexpr uint32_t last_element = UINT32_MAX >> 2u;
	if (state.profile.robust_buffer_access &&
	    (!offset.IsImmediate() || (offset.U32() >> 2u) + (mem.offset >> 2u) > last_element - 63u)) {
		element = EmitGlsl<GLSLstd450UMin, IR::Type::U32>(state, element,
		                                                  ConstantU32(state, last_element));
	}
	const auto condition = EmitStorageElementInBounds(state, access, element);
	ctx.Define(inst, EmitValueOrZeroIfCondition(state, condition, [&]() {
		           return EmitNative<spv::OpLoad, IR::Type::U32>(
		               state, EmitMemoryElementPointer(state, access, element));
	           }));
}

uint32_t EmitLoadDeviceAddressU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	return EmitValueOrZeroIfCondition(state, ctx.Arg(inst, 3), [&]() {
		const auto address = Binary(
		    state, spv::OpIAdd, TypeU64(state), PackU64(state, ctx.Arg(inst, 0), ctx.Arg(inst, 1)),
		    Unary(state, spv::OpUConvert, TypeU64(state), ctx.Arg(inst, 2)));
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpLoad, TypeU32(state), value,
		    Unary(state, spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), address),
		    spv::MemoryAccessAlignedMask, 4u);
		return value;
	});
}

uint32_t EmitGetBufferResourceIndex(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto resource_index = inst.Flags<uint32_t>();
	const auto& bindings = ctx.state.program.bindings;
	if (bindings.dynamic_indirect_buffers) {
		const auto root = std::ranges::find_if(bindings.indirect_buffer_roots,
		                                       [&](const IR::IndirectBufferRoot& entry) {
			                                       return entry.resource == resource_index;
		                                       });
		if (root == bindings.indirect_buffer_roots.end()) {
			return ConstantU32(ctx.state, 0u);
		}
		const auto mapping = EmitShaderDataDwordLoad(
		    ctx.state, bindings.IndirectRootMappingDword() + root->mapping_dword);
		return EmitIndirectResourceIndexDynamic(ctx.state, ctx.Arg(inst, 0), mapping,
		                                        root->binary_search ? 17u : 0u, UINT32_MAX);
	}
	const auto& buffer = ctx.state.program.info.buffers.at(resource_index);
	if (buffer.indirect_resources.empty()) return ConstantU32(ctx.state, 0u);
	return EmitIndirectResourceIndex(ctx.state, ctx.Arg(inst, 0), buffer.indirect_mapping_offset,
	                                 buffer.indirect_search_iterations, UINT32_MAX);
}

void EmitLoadMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op  = inst.GetOpcode();
	const auto& mem = ctx.Memory(inst);
	if (op == IR::ValueOpcode::LoadAddressU32 && mem.planning_only) return;
	const auto buffer_components = IR::BufferComponentCount(op);
	const auto shared_components = IR::SharedComponentCount(op);
	const auto address_info      = IR::AddressOpcodeInfoOf(op);
	uint32_t   value;
	if (mem.kind == IR::ResourceKind::FlatLocal)
		value = LoadLocalFlat(ctx, inst);
	else if (buffer_components > 1u ||
	         (buffer_components == 1u && mem.kind == IR::ResourceKind::IndirectBuffer) ||
	         (op == IR::ValueOpcode::LoadBufferU32 && mem.formatted))
		value = LoadBuffer(ctx, inst, buffer_components);
	else if (shared_components > 1u)
		value = LoadWideShared(ctx, inst, shared_components);
	else if (mem.kind == IR::ResourceKind::ScalarAddress)
		value = LoadBdaDword(ctx, GuestAddress(ctx, inst, mem));
	else if (address_info.access == IR::AddressAccess::Read &&
	         mem.kind != IR::ResourceKind::Scratch)
		value = LoadBda(ctx, GuestAddress(ctx, inst, mem), ctx.Arg(inst, inst.NumArgs() - 1),
		                address_info.data_bits, mem.coherent);
	else if (inst.GetType() == IR::Type::U8)
		value = LoadSubword(ctx, inst, mem, 8);
	else if (inst.GetType() == IR::Type::U16)
		value = LoadSubword(ctx, inst, mem, 16);
	else
		value = LoadWord(ctx, inst, mem);
	ctx.Define(inst, value);
}

void EmitStoreMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op                = inst.GetOpcode();
	const auto& mem               = ctx.Memory(inst);
	const auto  buffer_components = IR::BufferComponentCount(op);
	const auto  shared_components = IR::SharedComponentCount(op);
	const auto  type              = inst.Arg(inst.NumArgs() - 2).GetType();
	if (mem.kind == IR::ResourceKind::FlatLocal)
		StoreLocalFlat(ctx, inst);
	else if (buffer_components > 1u)
		StoreBuffer(ctx, inst, buffer_components);
	else if (shared_components > 1u)
		StoreWideShared(ctx, inst, shared_components);
	else if (op == IR::ValueOpcode::StoreBufferU32 && mem.formatted)
		StoreBuffer(ctx, inst, 1u);
	else if (type == IR::Type::U8)
		StoreSubword(ctx, inst, mem, 8);
	else if (type == IR::Type::U16)
		StoreSubword(ctx, inst, mem, 16);
	else
		StoreWord(ctx, inst, mem);
}

uint32_t EmitSharedIncDec(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto replacement =
	    inst.GetOpcode() == IR::ValueOpcode::SharedAtomicInc32 ? AtomicIncrement : AtomicDecrement;
	return EmitAtomicUpdate(ctx, inst, ctx.Memory(inst), replacement);
}

void EmitSharedAtomicMaskedOr32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto keep = Unary(ctx.state, spv::OpNot, TypeU32(ctx.state), ctx.Arg(inst, 1));
	EmitAtomicUpdate(ctx, inst, ctx.Memory(inst),
	                 [keep](EmitterState& state, uint32_t old, uint32_t value) {
		                 return Binary(state, spv::OpBitwiseOr, TypeU32(state),
		                               Binary(state, spv::OpBitwiseAnd, TypeU32(state), old, keep), value);
	                 });
}

uint32_t EmitSwizzleU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	state.builder.AddFunction(spv::OpStore, ctx.scratch_u32_variable, ctx.Arg(inst, 0));
	const auto source = EmitNative<spv::OpLoad, IR::Type::U32>(state, ctx.scratch_u32_variable);
	const auto target = EmitDsSwizzleTargetLane(state, EmitSubgroupLocalInvocationId(state),
	                                            inst.Arg(1).IsImmediate() ? inst.Arg(1).U32() : 0);
	return EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2));
}

uint32_t EmitPermuteU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state   = ctx.state;
	auto       lane    = EmitSubgroupLocalInvocationId(state);
	if (state.lane_count == 2) {
		lane = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane, ConstantU32(state, 31));
	}
	const auto address = ctx.Arg(inst, 1);
	const auto word    = Binary(state, spv::OpShiftRightLogical, TypeU32(state), lane,
	                            ConstantU32(state, 5));
	const auto ballot_word = [&](uint32_t predicate) {
		const auto ballot = state.builder.AllocateId();
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
		                          ConstantU32(state, spv::ScopeSubgroup), predicate);
		state.builder.AddFunction(spv::OpVectorExtractDynamic, TypeU32(state), result, ballot, word);
		return result;
	};
	// RDNA2 permutes independently within each 32-lane half. Intersect the source
	// address bit ballots to find this destination's enabled writers without LDS.
	auto writers = ballot_word(ctx.Arg(inst, 2));
	for (uint32_t bit = 0; bit < 5; ++bit) {
		const auto address_bit = Binary(state, spv::OpBitwiseAnd, TypeU32(state), address,
		                                ConstantU32(state, 1u << (bit + 2)));
		const auto mask = ballot_word(Binary(state, spv::OpINotEqual, TypeBool(state),
		                                      address_bit, ConstantU32(state, 0)));
		const auto lane_bit = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane,
		                             ConstantU32(state, 1u << bit));
		const auto selected = Select(
		    state, TypeU32(state),
		    Binary(state, spv::OpINotEqual, TypeBool(state), lane_bit, ConstantU32(state, 0)),
		    mask, Unary(state, spv::OpNot, TypeU32(state), mask));
		writers = Binary(state, spv::OpBitwiseAnd, TypeU32(state), writers, selected);
	}
	const auto active = Binary(state, spv::OpINotEqual, TypeBool(state), writers,
	                           ConstantU32(state, 0));
	const auto base = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane,
	                         ConstantU32(state, ~31u));
	const auto source = Select(state, TypeU32(state), active,
	                           Binary(state, spv::OpBitwiseOr, TypeU32(state), base,
	                                  EmitFindUMsb32(state, writers)), lane);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), ctx.Arg(inst, 0), source);
	return Select(state, TypeU32(state), active, result, ConstantU32(state, 0));
}

uint32_t EmitBpermuteU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto source = ctx.Arg(inst, 0);
	const auto index  = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                           Binary(state, spv::OpShiftRightLogical, TypeU32(state),
	                                  ctx.Arg(inst, 1), ConstantU32(state, 2)),
	                           ConstantU32(state, 31));
	const auto base   = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                           EmitSubgroupLocalInvocationId(state), ConstantU32(state, ~31u));
	const auto target = Binary(state, spv::OpBitwiseOr, TypeU32(state), base, index);
	return EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2));
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
