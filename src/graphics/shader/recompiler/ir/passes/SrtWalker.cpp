#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include "common/abi.h"
#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <deque>
#include <map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

SrtRuntime SrtReadCapture::ObservedRuntime() {
	auto runtime = m_source;
	runtime.userdata = this;
	runtime.read_specialization_memory = m_source.read_specialization_memory != nullptr
	                                         ? Read : nullptr;
	return runtime;
}

bool SrtReadCapture::Read(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (!capture.m_source.read_specialization_memory(capture.m_source.userdata, address, values)) {
		return false;
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

namespace {

uint32_t VectorBufferReadDwords(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::LoadBufferU32: return 1u;
		case ValueOpcode::LoadBufferU32x2: return 2u;
		case ValueOpcode::LoadBufferU32x3: return 3u;
		case ValueOpcode::LoadBufferU32x4: return 4u;
		default: return 0u;
	}
}

uint32_t CompositeExtractDwords(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::CompositeExtractU32x2: return 2u;
		case ValueOpcode::CompositeExtractU32x3: return 3u;
		case ValueOpcode::CompositeExtractU32x4: return 4u;
		default: return 0u;
	}
}

ValueOpcode CompositeConstructForDwords(uint32_t dwords) {
	switch (dwords) {
		case 2u: return ValueOpcode::CompositeConstructU32x2;
		case 3u: return ValueOpcode::CompositeConstructU32x3;
		case 4u: return ValueOpcode::CompositeConstructU32x4;
		default: return ValueOpcode::Void;
	}
}

bool UnpackReadNode(const Inst& node_inst, const Inst*& read_inst, uint32_t& component) {
	const auto op = node_inst.GetOpcode();
	if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
	    op == ValueOpcode::LoadBufferU32) {
		read_inst = &node_inst;
		component = 0u;
		return true;
	}
	const auto dwords = CompositeExtractDwords(op);
	if (dwords == 0u || node_inst.NumArgs() != 2u) {
		return false;
	}
	const auto* src = node_inst.Arg(0).Resolve().TryInstruction();
	const auto  idx = node_inst.Arg(1).Resolve();
	if (src == nullptr || VectorBufferReadDwords(src->GetOpcode()) != dwords ||
	    !idx.IsImmediate() || idx.GetType() != Type::U32 || idx.U32() >= dwords) {
		return false;
	}
	read_inst = src;
	component = idx.U32();
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op     = inst.GetOpcode();
	const auto dwords = VectorBufferReadDwords(op);
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer && dwords == 0u) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (dwords == 0u) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	return values.stage == ShaderType::Compute && inst.NumArgs() == 5u &&
	       memory.kind == ResourceKind::Buffer && !memory.typed && !memory.formatted &&
	       !memory.coherent && memory.data_bits == 32u && memory.data_dwords == dwords &&
	       !memory.offen && inst.Arg(2).Resolve() == Value(0u) &&
	       static_cast<int32_t>(memory.offset) >= 0 && inst.Arg(4).GetType() == Type::U1;
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertS32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::CompositeConstructU32x3:
		case ValueOpcode::CompositeExtractU32x3:
		case ValueOpcode::CompositeConstructU32x4:
		case ValueOpcode::CompositeExtractU32x4:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::SMin32:
		case ValueOpcode::SMax32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::UGreaterThanEqual32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

bool ImmediateBits(Value value, uint64_t& bits) {
	switch (value.GetType()) {
		case Type::U1: bits = value.U1(); return true;
		case Type::U8: bits = value.U8(); return true;
		case Type::U16: bits = value.U16(); return true;
		case Type::U32: bits = value.U32(); return true;
		case Type::U64: bits = value.U64(); return true;
		case Type::F32: bits = std::bit_cast<uint32_t>(value.F32Value()); return true;
		default: return false;
	}
}

// Scratch layout in bytes: per descriptor source its eight DWORDs and a known DWORD, one
// activity byte per source, then per image, then one qword per walked value (all ones when
// unknown). The walker's value slots, reachability, states, read bases and captured addresses
// follow.
constexpr uint32_t DescriptorStride = 9u * sizeof(uint32_t);

uint32_t ActiveOffset(const ResourcePlan& plan) {
	return static_cast<uint32_t>(plan.descriptor_sources.size()) * DescriptorStride;
}

uint32_t OutputOffset(const ResourcePlan& plan) {
	const auto active = plan.descriptor_sources.size() + plan.info.images.size();
	return (ActiveOffset(plan) + static_cast<uint32_t>(active) + 7u) & ~7u;
}

const uint8_t* Scratch(const ResourcePlan& plan) {
	return reinterpret_cast<const uint8_t*>(plan.srt_scratch.data());
}

thread_local const Xbyak::CodeGenerator* t_running_walker = nullptr;

bool MaskImplies(Value mask, Value target) {
	target = target.Resolve();
	if (mask.IsEmpty() || target.IsEmpty() || target.IsImmediate()) return false;
	std::vector<Value>              pending {mask};
	std::unordered_set<const Inst*> visited;
	while (!pending.empty()) {
		const auto current = pending.back().Resolve();
		pending.pop_back();
		if (current == target) return true;
		const auto* inst = current.TryInstruction();
		if (inst == nullptr || !visited.insert(inst).second) continue;
		if (inst->GetOpcode() == ValueOpcode::ConditionRef && inst->NumArgs() == 1u) {
			pending.push_back(inst->Arg(0));
		} else if (inst->GetOpcode() == ValueOpcode::LogicalAnd && inst->NumArgs() == 2u) {
			pending.push_back(inst->Arg(1));
			pending.push_back(inst->Arg(0));
		}
	}
	return false;
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type,
	                          bool wave_uniform = false)
	    : m_program(program), m_type(type), m_wave_uniform(wave_uniform) {}

	bool Run(Value value) { return ValidateIterative(value); }

private:
	struct ValidateFrame {
		enum class Step : uint8_t {
			Enter,
			AfterReadConstSlot,
			AfterSelectAnyCheck,
			AfterSelectFalseArm,
			AfterPhiArm,
			AfterReadFirstLaneMask,
			AfterArgs,
		};

		Step             step             = Step::Enter;
		Value            value;
		const Inst*      inst             = nullptr;
		bool             require_uniform  = true;
		RuntimeValueType type             = RuntimeValueType::Integer;
		Value            saved_mask;
		bool             has_saved_mask   = false;
		size_t           saved_lane_reads = 0;
		size_t           arg_index        = 0;
		bool             in_visiting      = false;
		bool             tail_only        = false;
	};

	std::unordered_set<const Inst*>& VisitingSet(RuntimeValueType type) {
		return type == RuntimeValueType::Any ? m_visiting_any : m_visiting;
	}

	std::unordered_set<const Inst*>& ValidatedSet(RuntimeValueType type, bool require_uniform) {
		if (type == RuntimeValueType::Any) {
			return require_uniform ? m_validated_any_uniform : m_validated_any_deps;
		}
		return require_uniform ? m_validated_uniform : m_validated_dependencies;
	}

	bool ValidateIterative(Value root) {
		std::vector<ValidateFrame> stack;
		stack.push_back(ValidateFrame {
		    .step            = ValidateFrame::Step::Enter,
		    .value           = root,
		    .require_uniform = true,
		    .type            = m_type,
		});
		bool last_result = false;

		const auto finish_frame = [&](ValidateFrame& frame, bool valid) {
			if (frame.has_saved_mask) {
				m_active_mask = frame.saved_mask;
			}
			if (frame.in_visiting) {
				VisitingSet(frame.type).erase(frame.inst);
				const bool cacheable = !frame.require_uniform || m_active_mask.IsEmpty();
				if (valid && cacheable &&
				    (!frame.require_uniform || frame.saved_lane_reads == m_lane_reads)) {
					ValidatedSet(frame.type, frame.require_uniform).insert(frame.inst);
				}
			}
			last_result = valid;
			stack.pop_back();
		};

		const auto push_child = [&](Value v, bool req_unif, RuntimeValueType ty) {
			stack.push_back(ValidateFrame {
			    .step            = ValidateFrame::Step::Enter,
			    .value           = v,
			    .require_uniform = req_unif,
			    .type            = ty,
			});
		};

		while (!stack.empty()) {
			auto& frame = stack.back();
			if (frame.tail_only) {
				finish_frame(frame, last_result);
				continue;
			}
			if (frame.step == ValidateFrame::Step::Enter) {
				const auto value = frame.value.Resolve();
				// Host floating-point evaluation does not model shader rounding/denormal modes.
				if (frame.type != RuntimeValueType::Any &&
				    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
					finish_frame(frame, false);
					continue;
				}
				const auto* inst = value.TryInstruction();
				if (inst == nullptr) {
					if (!frame.require_uniform) {
						finish_frame(frame, true);
						continue;
					}
					bool ok = false;
					switch (value.GetType()) {
						case Type::U1:
						case Type::U8:
						case Type::U16:
						case Type::U32:
						case Type::U64:
						case Type::F32: ok = true; break;
						default: break;
					}
					finish_frame(frame, ok);
					continue;
				}
				frame.value = value;
				frame.inst  = inst;
				if (frame.require_uniform && MaskImplies(m_active_mask, value)) {
					finish_frame(frame, true);
					continue;
				}
				const bool cacheable = !frame.require_uniform || m_active_mask.IsEmpty();
				if (cacheable && ValidatedSet(frame.type, frame.require_uniform).contains(inst)) {
					finish_frame(frame, true);
					continue;
				}
				if (!VisitingSet(frame.type).insert(inst).second) {
					finish_frame(frame, !frame.require_uniform);
					continue;
				}
				frame.in_visiting      = true;
				frame.saved_lane_reads = m_lane_reads;

				const auto op = inst->GetOpcode();
				if (frame.type == RuntimeValueType::ImmutableInteger &&
				    (op == ValueOpcode::ReadConst || BufferAccessOf(op) != BufferAccess::None ||
				     AddressOpcodeInfoOf(op).access != AddressAccess::None ||
				     ImageOpcodeInfoOf(op).access != ImageAccess::None ||
				     SharedAccessOf(op) != SharedAccess::None)) {
					finish_frame(frame, false);
					continue;
				}
				if (frame.type == RuntimeValueType::ScalarInteger) {
					const bool scalar_read =
					    op == ValueOpcode::LoadAddressU32 &&
					    inst->Flags<MemoryFlags>().index < m_program.memory_info.size() &&
					    m_program.memory_info[inst->Flags<MemoryFlags>().index].kind ==
					        ResourceKind::ScalarAddress;
					if (!scalar_read &&
					    (BufferAccessOf(op) != BufferAccess::None ||
					     AddressOpcodeInfoOf(op).access != AddressAccess::None ||
					     ImageOpcodeInfoOf(op).access != ImageAccess::None ||
					     SharedAccessOf(op) != SharedAccess::None)) {
						finish_frame(frame, false);
						continue;
					}
				}
				if (op == ValueOpcode::ReadConst) {
					const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
					if (inst->NumArgs() != 2 ||
					    inst->Arg(0).Resolve().TryInstruction() == nullptr ||
					    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
					        ValueOpcode::GetSrtResource ||
					    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
					    slot.U32() >= m_program.srt_reads.size()) {
						finish_frame(frame, false);
						continue;
					}
					if (frame.type != RuntimeValueType::Any) {
						frame.saved_mask     = m_active_mask;
						frame.has_saved_mask = true;
						m_active_mask        = {};
						frame.step           = ValidateFrame::Step::AfterReadConstSlot;
						push_child(m_program.srt_reads[slot.U32()].value, true, frame.type);
						continue;
					}
				}
				if (!frame.require_uniform) {
					if (inst->NumArgs() == 0) {
						finish_frame(frame, true);
						continue;
					}
					frame.step      = ValidateFrame::Step::AfterArgs;
					frame.arg_index = 1;
					push_child(inst->Arg(0), false, frame.type);
					continue;
				}
				if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
				    MaskImplies(m_active_mask, inst->Arg(0))) {
					if (inst->Arg(0).Resolve() == m_active_mask) {
						if (frame.type != RuntimeValueType::Any) {
							frame.step = ValidateFrame::Step::AfterSelectFalseArm;
							push_child(inst->Arg(2), false, frame.type);
						} else {
							frame.tail_only = true;
							push_child(inst->Arg(1), true, frame.type);
						}
						continue;
					}
					frame.saved_mask     = m_active_mask;
					frame.has_saved_mask = true;
					m_active_mask        = {};
					frame.step           = ValidateFrame::Step::AfterSelectAnyCheck;
					push_child(inst->Arg(0), true, RuntimeValueType::Any);
					continue;
				}
				if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU32 ||
				    op == ValueOpcode::Void) {
					finish_frame(frame, false);
					continue;
				}
				if (op == ValueOpcode::GetUserData) {
					if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
						finish_frame(frame, false);
						continue;
					}
					const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
					finish_frame(frame, reg >= m_program.user_data_base &&
					                        reg - m_program.user_data_base < m_program.user_data_count);
					continue;
				}
				if (op == ValueOpcode::GetShaderBase) {
					finish_frame(frame, inst->NumArgs() == 0);
					continue;
				}
				if (m_wave_uniform && op == ValueOpcode::GetBuiltin) {
					const auto axis = inst->NumArgs() == 2u ? inst->Arg(1).Resolve() : Value {};
					finish_frame(
					    frame,
					    m_program.stage == ShaderType::Compute && inst->NumArgs() == 2u &&
					        inst->Arg(0) ==
					            Value(static_cast<uint32_t>(StageInputKind::WorkgroupId)) &&
					        axis.IsImmediate() && axis.GetType() == Type::U32 && axis.U32() < 3u);
					continue;
				}
				if (op == ValueOpcode::Phi) {
					frame.step      = ValidateFrame::Step::AfterPhiArm;
					frame.arg_index = 0;
					last_result     = true;
					continue;
				}
				if (op == ValueOpcode::ReadFirstLane) {
					++m_lane_reads;
					if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
					    inst->Arg(1).GetType() != Type::U1) {
						finish_frame(frame, false);
						continue;
					}
					if (frame.type != RuntimeValueType::Any) {
						frame.step = ValidateFrame::Step::AfterReadFirstLaneMask;
						push_child(inst->Arg(1), false, frame.type);
						continue;
					}
					frame.saved_mask     = m_active_mask;
					frame.has_saved_mask = true;
					m_active_mask        = inst->Arg(1).Resolve();
					frame.tail_only      = true;
					push_child(inst->Arg(0), true, frame.type);
					continue;
				}
				if (op == ValueOpcode::GetSrtResource) {
					finish_frame(frame, inst->NumArgs() == 0);
					continue;
				}
				const auto vector_dwords = VectorBufferReadDwords(op);
				if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
				    vector_dwords != 0u) {
					const auto  expected = op == ValueOpcode::LoadAddressU32
					                           ? ValueOpcode::GetAddressResource
					                           : ValueOpcode::GetBufferResource;
					const auto* handle =
					    inst->NumArgs() != 0 ? inst->Arg(0).Resolve().TryInstruction() : nullptr;
					if (!IsRawRead(m_program, *inst) || handle == nullptr ||
					    handle->GetOpcode() != expected) {
						finish_frame(frame, false);
						continue;
					}
					if (vector_dwords != 0u) {
						frame.step      = ValidateFrame::Step::AfterArgs;
						frame.arg_index = 1;
						push_child(inst->Arg(0), true, frame.type);
						continue;
					}
				} else if (op == ValueOpcode::CompositeExtractU64) {
					const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
					if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
						finish_frame(frame, false);
						continue;
					}
				} else if (const auto dwords = CompositeExtractDwords(op); dwords != 0u) {
					const auto* source =
					    inst->NumArgs() == 2 ? inst->Arg(0).Resolve().TryInstruction() : nullptr;
					const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
					if (source == nullptr || !index.IsImmediate() ||
					    index.GetType() != Type::U32 || index.U32() >= dwords ||
					    (source->GetOpcode() != CompositeConstructForDwords(dwords) &&
					     VectorBufferReadDwords(source->GetOpcode()) != dwords)) {
						finish_frame(frame, false);
						continue;
					}
				}
				if (IsDescriptorHandle(op)) {
					size_t expected = 4u;
					if (op == ValueOpcode::GetImageResource) {
						expected = 8u;
					} else if (op == ValueOpcode::GetAddressResource) {
						expected = 2u;
					}
					if (inst->NumArgs() != expected) {
						finish_frame(frame, false);
						continue;
					}
				} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
				           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
					finish_frame(frame, false);
					continue;
				}
				if (inst->NumArgs() == 0) {
					finish_frame(frame, true);
					continue;
				}
				frame.step      = ValidateFrame::Step::AfterArgs;
				frame.arg_index = 1;
				push_child(inst->Arg(0), true, frame.type);
				continue;
			}

			switch (frame.step) {
				case ValidateFrame::Step::AfterReadConstSlot: {
					m_active_mask        = frame.saved_mask;
					frame.has_saved_mask = false;
					if (!last_result) {
						finish_frame(frame, false);
						break;
					}
					if (frame.inst->NumArgs() == 0) {
						finish_frame(frame, true);
						break;
					}
					frame.step      = ValidateFrame::Step::AfterArgs;
					frame.arg_index = 1;
					push_child(frame.inst->Arg(0), true, frame.type);
					break;
				}
				case ValidateFrame::Step::AfterSelectAnyCheck: {
					m_active_mask        = frame.saved_mask;
					frame.has_saved_mask = false;
					if (!last_result) {
						// Condition is not unmasked uniform; active mask proves Arg(1) runs.
						if (frame.type != RuntimeValueType::Any) {
							frame.step = ValidateFrame::Step::AfterSelectFalseArm;
							push_child(frame.inst->Arg(2), false, frame.type);
						} else {
							frame.step      = ValidateFrame::Step::Enter;
							frame.tail_only = true;
							push_child(frame.inst->Arg(1), true, frame.type);
						}
						break;
					}
					frame.step      = ValidateFrame::Step::AfterArgs;
					frame.arg_index = 1;
					push_child(frame.inst->Arg(0), true, frame.type);
					break;
				}
				case ValidateFrame::Step::AfterSelectFalseArm: {
					if (!last_result) {
						finish_frame(frame, false);
						break;
					}
					frame.step      = ValidateFrame::Step::Enter;
					frame.tail_only = true;
					push_child(frame.inst->Arg(1), true, frame.type);
					break;
				}
				case ValidateFrame::Step::AfterPhiArm: {
					if (!last_result) {
						finish_frame(frame, false);
						break;
					}
					const auto* inst  = frame.inst;
					const auto* merge = inst->Parent();
					bool        pushed = false;
					if (frame.type != RuntimeValueType::Any) {
						while (frame.arg_index < inst->NumArgs()) {
							const size_t idx = frame.arg_index++;
							if (merge != nullptr && idx < inst->NumPhiBlocks() &&
							    IsUnreachablePhiEdge(inst->PhiBlock(idx), merge)) {
								continue;
							}
							push_child(inst->Arg(idx), false, frame.type);
							pushed = true;
							break;
						}
					}
					if (pushed) break;
					const auto invariant = ResolveInvariantPhi(m_program, frame.value);
					if (invariant.IsEmpty()) {
						finish_frame(frame, false);
						break;
					}
					frame.step      = ValidateFrame::Step::Enter;
					frame.tail_only = true;
					push_child(invariant, true, frame.type);
					break;
				}
				case ValidateFrame::Step::AfterReadFirstLaneMask: {
					if (!last_result) {
						finish_frame(frame, false);
						break;
					}
					frame.saved_mask     = m_active_mask;
					frame.has_saved_mask = true;
					m_active_mask        = frame.inst->Arg(1).Resolve();
					frame.step           = ValidateFrame::Step::Enter;
					frame.tail_only      = true;
					push_child(frame.inst->Arg(0), true, frame.type);
					break;
				}
				case ValidateFrame::Step::AfterArgs: {
					if (!last_result) {
						finish_frame(frame, false);
						break;
					}
					const auto* inst          = frame.inst;
					const bool  vector_dwords = VectorBufferReadDwords(inst->GetOpcode()) != 0u;
					if (vector_dwords && frame.arg_index == 4u) {
						const auto guard = inst->Arg(4).Resolve();
						if (!m_active_mask.IsEmpty() && guard == m_active_mask) {
							finish_frame(frame, true);
							break;
						}
					}
					if (frame.arg_index >= inst->NumArgs()) {
						finish_frame(frame, true);
						break;
					}
					const size_t idx = frame.arg_index++;
					push_child(inst->Arg(idx), frame.require_uniform, frame.type);
					break;
				}
				default: finish_frame(frame, false); break;
			}
		}
		return last_result;
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	bool                            m_wave_uniform;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_visiting_any;
	std::unordered_set<const Inst*> m_validated_dependencies;
	std::unordered_set<const Inst*> m_validated_uniform;
	std::unordered_set<const Inst*> m_validated_any_deps;
	std::unordered_set<const Inst*> m_validated_any_uniform;
	size_t                          m_lane_reads = 0;
};

// Emits one x86 function per plan. Each value, per EXEC-mask context, is a memoized thunk with a
// state byte (0 unevaluated, 1 unknown or in progress, 2 known, 3 a control Phi holding its first
// value) and a qword slot; a reference is `call thunk; cmp state, 2`. A value the walk body already
// referenced is tested first and its call runs out of line. Operands that are unknown, constant or
// plain user data are folded at emission. Registers: rdi user data, rsi flat buffer, rdx scratch
// plus m_bias, r8 capture cursor.
class WalkerCompiler {
public:
	WalkerCompiler(ResourcePlan& plan, Xbyak::CodeGenerator& code): m_plan(plan), m_code(code) {}

	void Compile(std::span<const Value> values);

private:
	enum class Kind : uint8_t { Unknown, Constant, UserData, ShaderBase, Node };
	struct Operand {
		Kind     kind  = Kind::Unknown;
		uint64_t value = 0;

		bool operator==(const Operand& other) const = default;
	};
	// masked: the operand depends on its context's EXEC mask, so it is not the root context's.
	struct Ref {
		Operand operand;
		bool    masked = false;
	};
	struct Node {
		const Inst*          inst    = nullptr;
		uint32_t             context = 0;
		Operand              result;
		std::vector<Operand> args;
		uint32_t             slot = 0, state = 0;
		Xbyak::Label         label;
		bool                 pending = true, referenced = false, masked = false, called = false;
	};
	// An out-of-line thunk call for a walk body reference to a value the body referenced before.
	struct Stub {
		Operand             operand;
		const Xbyak::Label* unknown = nullptr;
		Xbyak::Label        entry, back;
	};

	static constexpr auto Near = Xbyak::CodeGenerator::T_NEAR;

	// The scratch address at offset; offsets m_reach to m_reach + 255 take 8-bit displacements.
	Xbyak::RegExp At(size_t offset) const { return m_code.rdx - m_bias + offset; }

	bool Thunk(uint32_t index) const {
		const auto& node = m_nodes[index];
		return node.referenced || node.result == Operand {Kind::Node, index};
	}
	static bool MayBeUnknown(const Operand& operand) {
		return operand.kind == Kind::Unknown || operand.kind == Kind::Node;
	}

	uint32_t Context(Value mask) {
		mask = mask.Resolve();
		const auto found = std::ranges::find(m_masks, mask);
		if (found != m_masks.end()) return static_cast<uint32_t>(found - m_masks.begin());
		m_masks.push_back(mask);
		return static_cast<uint32_t>(m_masks.size() - 1u);
	}

	void AddOutput(Value value, uint32_t gate);
	void Consume(std::vector<std::vector<uint32_t>>& gates);
	Ref Evaluate(Value value, uint32_t context);
	void Layout();
	void Gate(std::vector<uint32_t>& gates, const Xbyak::Label& skip);
	void Zero(uint32_t offset, uint32_t size);
	void Branch(const Operand& operand, const Xbyak::Label& unknown);
	void Load(const Xbyak::Reg64& reg, const Operand& operand);
	void Base(const Operand& low, const Operand& high);
	void Fetch(const Operand& operand, const Xbyak::Label& unknown);
	void Set(void (Xbyak::CodeGenerator::*condition)(const Xbyak::Operand&));
	void Store(const Node& node);
	void EmitNode(uint32_t index);
	void EmitRead(const Node& node);
	void EmitStrict(const Node& node);

	ResourcePlan&                                     m_plan;
	Xbyak::CodeGenerator&                             m_code;
	std::vector<Value>                                m_masks {Value {}};
	std::deque<Node>                                  m_nodes;
	std::map<std::pair<const Inst*, uint32_t>, uint32_t> m_memo;
	std::vector<Value>                                m_outputs;
	std::vector<std::vector<uint32_t>>                m_output_gates;
	// Per user-data base pair of a read, the scratch slot holding the base for the walk.
	std::map<std::pair<uint64_t, uint64_t>, uint32_t> m_bases;
	std::deque<Stub>                                  m_stubs;
	Xbyak::Label                                      m_ret;
	uint32_t m_shader_base = 0, m_reach = 0, m_states = 0, m_states_end = 0, m_bias = 0;
	uint32_t m_user_data = 0, m_captures = 0;
	bool     m_body = true;
};

void WalkerCompiler::AddOutput(Value value, uint32_t gate) {
	const auto* inst = value.Resolve().TryInstruction();
	if (inst == nullptr) return;
	auto index = inst->WalkOutput();
	if (index >= m_outputs.size() || m_outputs[index].TryInstruction() != inst) {
		index = static_cast<uint32_t>(m_outputs.size());
		inst->SetWalkOutput(index);
		m_outputs.push_back(value.Resolve());
		m_output_gates.emplace_back();
	}
	m_output_gates[index].push_back(gate);
}

// Mirrors MaterializeResources: each descriptor it reads back is gated by the activity of the
// resource root that needs it, and so are an indirect root's selector values.
void WalkerCompiler::Consume(std::vector<std::vector<uint32_t>>& gates) {
	const auto& sources = m_plan.descriptor_sources;
	const auto  consume = [&](uint32_t root) {
		if (root >= sources.size()) return;
		const auto& indirect = sources[root].indirect_descriptor;
		if (!indirect) {
			gates[root].push_back(root);
			return;
		}
		const auto add = [&](uint32_t source) {
			if (source < sources.size()) gates[source].push_back(root);
		};
		if (indirect->sources.empty()) {
			add(indirect->table_source);
			if (indirect->selector) {
				add(indirect->selector->source);
				if (indirect->selector->rows) add(indirect->selector->rows->source);
			}
		}
		for (const auto source: indirect->sources) add(source);
		AddOutput(indirect->key_count, root);
		AddOutput(indirect->selector_first, root);
		AddOutput(indirect->selector_mask, root);
	};
	for (const auto& buffer: m_plan.info.buffers) consume(buffer.source);
	for (const auto& image: m_plan.info.images) consume(image.source);
	for (const auto& sampler: m_plan.info.samplers) {
		const auto* indirect = sampler.source < sources.size() && sources[sampler.source].indirect_descriptor
		    ? &*sources[sampler.source].indirect_descriptor : nullptr;
		if (indirect == nullptr || indirect->sources.empty()) {
			consume(sampler.source);
		} else if (indirect->sources[0] < sources.size()) {
			gates[indirect->sources[0]].push_back(indirect->sources[0]);
		}
	}
	const auto& fill = m_plan.uniform_fill;
	if (fill.fill.words != 0) {
		AddOutput(fill.pattern_size, UINT32_MAX);
		AddOutput(fill.element_count, UINT32_MAX);
		for (const auto value: fill.values) AddOutput(value, UINT32_MAX);
	}
}

WalkerCompiler::Ref WalkerCompiler::Evaluate(Value root, uint32_t root_context) {
	enum class FrameStep : uint8_t {
		Enter,
		DispatchNode,
		ReturnForceMasked,
		ReturnUnmaskedOperand,
		AfterMaskImpliesUnmasked,
		AfterSelectCondCheck,
		AfterUnmaskedRetry,
		AnalyzePhiArg,
		AnalyzeSelectCond,
		AnalyzeSelectChosen,
		AnalyzeSelectTrue,
		AnalyzeSelectFalse,
		AnalyzeUCmpBound,
		AnalyzeUCmpOther,
		AnalyzeLogicalFirst,
		AnalyzeLogicalSecond,
		AnalyzeLogicalRefined,
		AnalyzeDefaultArg,
		AnalyzeReadGuard,
		AnalyzeReadWord,
	};

	struct Frame {
		FrameStep            step       = FrameStep::Enter;
		Value                value;
		uint32_t             context    = 0;
		uint32_t             node_index = UINT32_MAX;
		size_t               arg_index  = 0;
		bool                 known      = false;
		bool                 failed     = false;
		bool                 read_buf   = false;
		bool                 read_vec   = false;
		Operand              op_a {};
		Operand              op_b {};
		std::array<Value, 7> read_words {};
		uint8_t              read_count = 0;
	};

	const auto constant = [](uint64_t val) { return Operand {Kind::Constant, val}; };

	std::vector<Frame> stack;
	stack.push_back(Frame {.step = FrameStep::Enter, .value = root, .context = root_context});
	Ref last_ref {};

	const auto finish_frame = [&](Ref ref) {
		last_ref = ref;
		stack.pop_back();
	};

	const auto finish_node = [&](Frame& frame, Operand op_res) {
		auto& node   = m_nodes[frame.node_index];
		node.result  = op_res;
		node.pending = false;
		if (node.context != 0 && !node.masked && !node.referenced) {
			frame.step = FrameStep::AfterUnmaskedRetry;
			stack.push_back(
			    Frame {.step = FrameStep::Enter, .value = frame.value, .context = 0u});
			return;
		}
		finish_frame({node.result, node.masked});
	};

	const auto push_eval = [&](Value v, uint32_t ctx) {
		stack.push_back(Frame {.step = FrameStep::Enter, .value = v, .context = ctx});
	};

	while (!stack.empty()) {
		auto& frame = stack.back();
		switch (frame.step) {
			case FrameStep::ReturnForceMasked:
				finish_frame({last_ref.operand, true});
				break;
			case FrameStep::ReturnUnmaskedOperand:
				finish_frame({last_ref.operand, false});
				break;
			case FrameStep::AfterMaskImpliesUnmasked:
				if (last_ref.operand.kind != Kind::Unknown) {
					finish_frame(last_ref);
				} else {
					finish_frame({{Kind::Constant, 1u}, true});
				}
				break;
			case FrameStep::AfterUnmaskedRetry: {
				auto& node  = m_nodes[frame.node_index];
				node.result = last_ref.operand;
				finish_frame({node.result, false});
				break;
			}
			case FrameStep::Enter:
			case FrameStep::AfterSelectCondCheck: {
				const bool skip_select_check = frame.step == FrameStep::AfterSelectCondCheck;
				bool       done_or_pushed    = false;
				for (;;) {
					if (!skip_select_check || frame.step == FrameStep::Enter) {
						frame.value = frame.value.Resolve();
						if (frame.value.IsEmpty()) {
							finish_frame({});
							done_or_pushed = true;
							break;
						}
						if (frame.value.IsImmediate()) {
							uint64_t bits = 0;
							finish_frame(ImmediateBits(frame.value, bits)
							                 ? Ref {{Kind::Constant, bits}}
							                 : Ref {});
							done_or_pushed = true;
							break;
						}
						const auto mask = m_masks[frame.context];
						if (frame.context != 0 && frame.value == mask) {
							finish_frame({{Kind::Constant, 1u}, true});
							done_or_pushed = true;
							break;
						}
						if (frame.context != 0 && MaskImplies(mask, frame.value)) {
							frame.step = FrameStep::AfterMaskImpliesUnmasked;
							push_eval(frame.value, 0u);
							done_or_pushed = true;
							break;
						}
					} else if (last_ref.operand.kind == Kind::Unknown) {
						const auto* select_inst = frame.value.TryInstruction();
						frame.step              = FrameStep::ReturnForceMasked;
						push_eval(select_inst->Arg(1), frame.context);
						done_or_pushed = true;
						break;
					}

					const auto* inst = frame.value.TryInstruction();
					if (inst == nullptr) {
						finish_frame({});
						done_or_pushed = true;
						break;
					}
					const auto op   = inst->GetOpcode();
					const auto mask = m_masks[frame.context];
					if (!skip_select_check && frame.context != 0 && IsRuntimeSelect(op) &&
					    inst->NumArgs() == 3 && MaskImplies(mask, inst->Arg(0))) {
						if (inst->Arg(0).Resolve() == mask) {
							frame.step = FrameStep::ReturnForceMasked;
							push_eval(inst->Arg(1), frame.context);
							done_or_pushed = true;
							break;
						}
						frame.step = FrameStep::AfterSelectCondCheck;
						push_eval(inst->Arg(0), 0u);
						done_or_pushed = true;
						break;
					}

					// Now dispatch on op / memoize (either directly or after select check).
					bool tail_chase = false;
					switch (op) {
						case ValueOpcode::BitCastU32F32:
						case ValueOpcode::BitCastF32U32:
						case ValueOpcode::ConditionRef:
							frame.value = inst->Arg(0);
							frame.step  = FrameStep::Enter;
							tail_chase  = true;
							break;
						case ValueOpcode::CompositeExtractU32x2:
						case ValueOpcode::CompositeExtractU32x3:
						case ValueOpcode::CompositeExtractU32x4: {
							const auto dwords = CompositeExtractDwords(op);
							const auto* source =
							    inst->NumArgs() == 2 ? inst->Arg(0).Resolve().TryInstruction() : nullptr;
							const auto index =
							    inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
							if (source == nullptr || !index.IsImmediate() ||
							    index.GetType() != Type::U32 || index.U32() >= dwords) {
								finish_frame({});
								done_or_pushed = true;
								break;
							}
							if (source->GetOpcode() == CompositeConstructForDwords(dwords)) {
								frame.value = source->Arg(index.U32());
								frame.step  = FrameStep::Enter;
								tail_chase  = true;
								break;
							}
							if (VectorBufferReadDwords(source->GetOpcode()) == dwords) {
								break;
							}
							finish_frame({});
							done_or_pushed = true;
							break;
						}
						case ValueOpcode::ReadFirstLane:
							frame.step = FrameStep::ReturnUnmaskedOperand;
							push_eval(inst->Arg(0), Context(inst->Arg(1)));
							done_or_pushed = true;
							break;
						case ValueOpcode::GetUserData: {
							const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
							if (reg < m_plan.user_data_base ||
							    reg - m_plan.user_data_base >= m_plan.user_data_count) {
								finish_frame({});
							} else {
								finish_frame({{Kind::UserData, reg - m_plan.user_data_base}});
							}
							done_or_pushed = true;
							break;
						}
						case ValueOpcode::GetShaderBase:
							finish_frame({{Kind::ShaderBase}});
							done_or_pushed = true;
							break;
						case ValueOpcode::Phi:
							if (!m_plan.control_phi_predecessors.contains(inst)) {
								const auto invariant = ResolveInvariantPhi(m_plan, frame.value);
								if (invariant.IsEmpty()) {
									finish_frame({});
									done_or_pushed = true;
									break;
								}
								frame.value = invariant;
								frame.step  = FrameStep::Enter;
								tail_chase  = true;
								break;
							}
							break;
						default: break;
					}
					if (done_or_pushed) break;
					if (tail_chase) {
						frame.step = FrameStep::Enter;
						break;
					}

					const auto [found, inserted] =
					    m_memo.try_emplace({inst, frame.context},
					                       static_cast<uint32_t>(m_nodes.size()));
					const auto index = found->second;
					if (!inserted) {
						auto& existing = m_nodes[index];
						if (!existing.pending) {
							finish_frame({existing.result, existing.masked});
						} else {
							existing.referenced = true;
							finish_frame({{Kind::Node, index}, frame.context != 0});
						}
						done_or_pushed = true;
						break;
					}
					auto& node   = m_nodes.emplace_back();
					node.inst    = inst;
					node.context = frame.context;
					frame.node_index = index;
					frame.step       = FrameStep::DispatchNode;
					done_or_pushed   = true;
					break;
				}
				(void)done_or_pushed;
				break;
			}
			case FrameStep::DispatchNode: {
				auto&       node    = m_nodes[frame.node_index];
				const auto& inst    = *node.inst;
				const auto  self    = Operand {Kind::Node, frame.node_index};
				const auto  op      = inst.GetOpcode();
				const auto  context = node.context;
				switch (op) {
					case ValueOpcode::Phi: {
						if (inst.NumArgs() == 0u) {
							finish_node(frame, Operand {});
							break;
						}
						frame.step      = FrameStep::AnalyzePhiArg;
						frame.arg_index = 0u;
						frame.known     = false;
						push_eval(inst.Arg(0), context);
						break;
					}
					case ValueOpcode::LoadAddressU32:
					case ValueOpcode::ReadConstBuffer:
					case ValueOpcode::LoadBufferU32:
					case ValueOpcode::CompositeExtractU32x2:
					case ValueOpcode::CompositeExtractU32x3:
					case ValueOpcode::CompositeExtractU32x4: {
						const Inst* read_inst = nullptr;
						uint32_t    component = 0u;
						if (!UnpackReadNode(inst, read_inst, component)) {
							finish_node(frame, Operand {});
							break;
						}
						const auto* handle = read_inst->Arg(0).Resolve().TryInstruction();
						if (!IsRawRead(m_plan, *read_inst) || handle == nullptr) {
							finish_node(frame, Operand {});
							break;
						}
						const bool vector = VectorBufferReadDwords(read_inst->GetOpcode()) != 0u;
						const bool buffer = read_inst->GetOpcode() != ValueOpcode::LoadAddressU32;
						const auto offset = static_cast<int32_t>(
						    m_plan.memory_info[read_inst->Flags<MemoryFlags>().index].offset);
						frame.read_vec = vector;
						frame.read_buf = buffer;
						frame.failed   = buffer && (handle->NumArgs() != 4u || offset < 0);
						if (vector) {
							frame.read_words = {handle->Arg(0), handle->Arg(1), handle->Arg(2),
							                    handle->Arg(3), read_inst->Arg(1),
							                    read_inst->Arg(2), read_inst->Arg(3)};
							frame.read_count = 7u;
							frame.step       = FrameStep::AnalyzeReadGuard;
							push_eval(read_inst->Arg(4), context);
						} else {
							frame.read_words[0] = handle->Arg(0);
							frame.read_words[1] = handle->Arg(1);
							frame.read_words[2] = read_inst->Arg(1);
							if (buffer && handle->NumArgs() == 4u) {
								frame.read_words[3] = handle->Arg(2);
								frame.read_words[4] = handle->Arg(3);
							}
							frame.read_count    = 3u;
							frame.arg_index     = 0u;
							frame.step          = FrameStep::AnalyzeReadWord;
							push_eval(frame.read_words[0], context);
						}
						break;
					}
					case ValueOpcode::SelectU1:
					case ValueOpcode::SelectU32:
					case ValueOpcode::SelectF32:
						frame.step = FrameStep::AnalyzeSelectCond;
						push_eval(inst.Arg(0), context);
						break;
					case ValueOpcode::ULessThan32:
					case ValueOpcode::ULessThanEqual32:
					case ValueOpcode::UGreaterThanEqual32: {
						const bool bound_right =
						    op == ValueOpcode::ULessThan32 || op == ValueOpcode::UGreaterThanEqual32;
						frame.step = FrameStep::AnalyzeUCmpBound;
						push_eval(inst.Arg(bound_right ? 1 : 0), context);
						break;
					}
					case ValueOpcode::LogicalOr:
					case ValueOpcode::LogicalAnd:
						frame.step = FrameStep::AnalyzeLogicalFirst;
						push_eval(inst.Arg(0), context);
						break;
					case ValueOpcode::CompositeExtractU64: {
						const auto component = inst.Arg(1).Resolve();
						if (!component.IsImmediate() || component.GetType() != Type::U32 ||
						    component.U32() >= 2u) {
							finish_node(frame, Operand {});
							break;
						}
						[[fallthrough]];
					}
					default: {
						if (!IsRuntimeUniformOp(op) ||
						    op == ValueOpcode::CompositeConstructU32x2 ||
						    op == ValueOpcode::CompositeConstructU32x3 ||
						    op == ValueOpcode::CompositeConstructU32x4) {
							finish_node(frame, Operand {});
							break;
						}
						if (inst.NumArgs() == 0u) {
							finish_node(frame, self);
							break;
						}
						frame.step      = FrameStep::AnalyzeDefaultArg;
						frame.arg_index = 0u;
						push_eval(inst.Arg(0), context);
						break;
					}
				}
				break;
			}
			case FrameStep::AnalyzePhiArg: {
				auto&       node = m_nodes[frame.node_index];
				const auto& inst = *node.inst;
				node.masked |= last_ref.masked;
				node.args.push_back(last_ref.operand);
				frame.known |= last_ref.operand.kind != Kind::Unknown;
				if (++frame.arg_index < inst.NumArgs()) {
					push_eval(inst.Arg(frame.arg_index), node.context);
					break;
				}
				finish_node(frame, frame.known ? Operand {Kind::Node, frame.node_index} : Operand {});
				break;
			}
			case FrameStep::AnalyzeReadGuard: {
				auto& node = m_nodes[frame.node_index];
				node.masked |= last_ref.masked;
				const auto guard = last_ref.operand;
				if (guard.kind == Kind::Unknown) {
					finish_node(frame, Operand {});
					break;
				}
				if (guard.kind == Kind::Constant && guard.value == 0u) {
					finish_node(frame, Operand {Kind::Constant, 0u});
					break;
				}
				node.args.push_back(guard);
				frame.arg_index = 0u;
				frame.step      = FrameStep::AnalyzeReadWord;
				push_eval(frame.read_words[0], node.context);
				break;
			}
			case FrameStep::AnalyzeReadWord: {
				auto& node = m_nodes[frame.node_index];
				node.masked |= last_ref.masked;
				node.args.push_back(last_ref.operand);
				frame.failed |= last_ref.operand.kind == Kind::Unknown;
				++frame.arg_index;
				if (!frame.read_vec && frame.arg_index == 3u && frame.read_buf && !frame.failed) {
					frame.read_count = 5u;
				}
				if (frame.arg_index < frame.read_count) {
					push_eval(frame.read_words[frame.arg_index], node.context);
					break;
				}
				if (!frame.failed) {
					finish_node(frame, Operand {Kind::Node, frame.node_index});
					break;
				}
				if (!frame.read_vec || node.args[0].kind == Kind::Constant) {
					finish_node(frame, Operand {});
					break;
				}
				node.args.resize(1u);
				finish_node(frame, Operand {Kind::Node, frame.node_index});
				break;
			}
			case FrameStep::AnalyzeSelectCond: {
				auto&       node = m_nodes[frame.node_index];
				const auto& inst = *node.inst;
				node.masked |= last_ref.masked;
				const auto predicate = last_ref.operand;
				if (predicate.kind == Kind::Unknown) {
					finish_node(frame, Operand {});
					break;
				}
				if (predicate.kind == Kind::Constant) {
					frame.step = FrameStep::AnalyzeSelectChosen;
					push_eval(inst.Arg(predicate.value != 0u ? 1 : 2), node.context);
					break;
				}
				frame.op_a = predicate;
				frame.step = FrameStep::AnalyzeSelectTrue;
				push_eval(inst.Arg(1), node.context);
				break;
			}
			case FrameStep::AnalyzeSelectChosen: {
				auto& node = m_nodes[frame.node_index];
				node.masked |= last_ref.masked;
				finish_node(frame, last_ref.operand);
				break;
			}
			case FrameStep::AnalyzeSelectTrue: {
				auto&       node = m_nodes[frame.node_index];
				const auto& inst = *node.inst;
				node.masked |= last_ref.masked;
				frame.op_b = last_ref.operand;
				frame.step = FrameStep::AnalyzeSelectFalse;
				push_eval(inst.Arg(2), node.context);
				break;
			}
			case FrameStep::AnalyzeSelectFalse: {
				auto& node = m_nodes[frame.node_index];
				node.masked |= last_ref.masked;
				node.args = {frame.op_a, frame.op_b, last_ref.operand};
				finish_node(frame, node.args[1].kind == Kind::Unknown &&
				                           node.args[2].kind == Kind::Unknown
				                       ? Operand {}
				                       : Operand {Kind::Node, frame.node_index});
				break;
			}
			case FrameStep::AnalyzeUCmpBound: {
				auto&       node        = m_nodes[frame.node_index];
				const auto& inst        = *node.inst;
				const auto  op          = inst.GetOpcode();
				const bool  less        = op == ValueOpcode::ULessThan32;
				const bool  bound_right = less || op == ValueOpcode::UGreaterThanEqual32;
				node.masked |= last_ref.masked;
				const auto bound = last_ref.operand;
				if (bound.kind == Kind::Unknown) {
					finish_node(frame, Operand {});
					break;
				}
				if (bound.kind == Kind::Constant && static_cast<uint32_t>(bound.value) == 0u) {
					finish_node(frame, constant(!less));
					break;
				}
				frame.op_a = bound;
				frame.step = FrameStep::AnalyzeUCmpOther;
				push_eval(inst.Arg(bound_right ? 0 : 1), node.context);
				break;
			}
			case FrameStep::AnalyzeUCmpOther: {
				auto& node = m_nodes[frame.node_index];
				node.masked |= last_ref.masked;
				node.args = {frame.op_a, last_ref.operand};
				finish_node(frame,
				            node.args[1].kind == Kind::Unknown && frame.op_a.kind == Kind::Constant
				                ? Operand {}
				                : Operand {Kind::Node, frame.node_index});
				break;
			}
			case FrameStep::AnalyzeLogicalFirst: {
				auto&       node        = m_nodes[frame.node_index];
				const auto& inst        = *node.inst;
				const bool  conjunction = inst.GetOpcode() == ValueOpcode::LogicalAnd;
				node.masked |= last_ref.masked;
				const auto a = last_ref.operand;
				if (a.kind == Kind::Constant && (a.value == 0u) == conjunction) {
					finish_node(frame, constant(!conjunction));
					break;
				}
				frame.op_a = a;
				frame.step = FrameStep::AnalyzeLogicalSecond;
				push_eval(inst.Arg(1), node.context);
				break;
			}
			case FrameStep::AnalyzeLogicalSecond: {
				auto&       node        = m_nodes[frame.node_index];
				const auto& inst        = *node.inst;
				const bool  conjunction = inst.GetOpcode() == ValueOpcode::LogicalAnd;
				const auto  decides     = [&](const Operand& x) {
					return x.kind == Kind::Constant && (x.value == 0u) == conjunction;
				};
				node.masked |= last_ref.masked;
				const auto a = frame.op_a;
				const auto b = last_ref.operand;
				if (decides(b)) {
					finish_node(frame, constant(!conjunction));
					break;
				}
				if (a.kind == Kind::Constant && b.kind == Kind::Constant) {
					finish_node(frame, constant(conjunction));
					break;
				}
				if (a.kind == Kind::Constant && b.kind == Kind::Unknown) {
					finish_node(frame, Operand {});
					break;
				}
				if (conjunction && MayBeUnknown(a)) {
					frame.op_b = b;
					frame.step = FrameStep::AnalyzeLogicalRefined;
					push_eval(inst.Arg(1), Context(inst.Arg(0)));
					break;
				}
				if (a.kind == Kind::Unknown && (b.kind == Kind::Unknown || b.kind == Kind::Constant)) {
					finish_node(frame, Operand {});
					break;
				}
				node.args = {a, b, Operand {}};
				finish_node(frame, Operand {Kind::Node, frame.node_index});
				break;
			}
			case FrameStep::AnalyzeLogicalRefined: {
				auto&      node    = m_nodes[frame.node_index];
				const auto a       = frame.op_a;
				const auto b       = frame.op_b;
				auto       refined = last_ref.operand;
				if (refined == b || (refined.kind == Kind::Constant && refined.value != 0u)) {
					refined = {};
				}
				if (a.kind == Kind::Unknown && refined.kind == Kind::Constant) {
					finish_node(frame, constant(0u));
					break;
				}
				if (a.kind == Kind::Unknown && refined.kind == Kind::Unknown &&
				    (b.kind == Kind::Unknown || b.kind == Kind::Constant)) {
					finish_node(frame, Operand {});
					break;
				}
				node.args = {a, b, refined};
				finish_node(frame, Operand {Kind::Node, frame.node_index});
				break;
			}
			case FrameStep::AnalyzeDefaultArg: {
				auto&       node = m_nodes[frame.node_index];
				const auto& inst = *node.inst;
				node.masked |= last_ref.masked;
				node.args.push_back(last_ref.operand);
				if (last_ref.operand.kind == Kind::Unknown) {
					finish_node(frame, Operand {});
					break;
				}
				if (++frame.arg_index < inst.NumArgs()) {
					push_eval(inst.Arg(frame.arg_index), node.context);
					break;
				}
				finish_node(frame, Operand {Kind::Node, frame.node_index});
				break;
			}
		}
	}
	return last_ref;
}

void WalkerCompiler::Layout() {
	uint32_t offset = OutputOffset(m_plan) + static_cast<uint32_t>(m_outputs.size()) * 8u;
	m_shader_base = offset;
	offset += 8u;
	for (uint32_t index = 0; index < m_nodes.size(); ++index) {
		if (!Thunk(index)) continue;
		auto& node = m_nodes[index];
		node.slot  = offset;
		offset += 8u;
		const auto op   = node.inst->GetOpcode();
		const bool read = op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer;
		if (read && node.result == Operand {Kind::Node, index} && node.args.size() > 2u &&
		    node.args[0].kind == Kind::UserData && node.args[1].kind == Kind::UserData) {
			m_bases.try_emplace({node.args[0].value, node.args[1].value}, 0u);
		}
	}
	m_reach = offset;
	m_bias  = m_reach + 128u;
	offset += static_cast<uint32_t>(m_plan.control_flow.size());
	m_states = offset;
	for (uint32_t index = 0; index < m_nodes.size(); ++index) {
		if (Thunk(index)) m_nodes[index].state = offset++;
	}
	m_states_end = offset;
	offset       = (offset + 7u) & ~7u;
	for (auto& [pair, slot]: m_bases) {
		slot = offset;
		offset += 8u;
	}
	// Captured addresses come last; EmitRead counts them.
	m_plan.srt_captures = offset;
}

// Falls through when any gating source is active. The list is not empty.
void WalkerCompiler::Gate(std::vector<uint32_t>& gates, const Xbyak::Label& skip) {
	auto& c = m_code;
	std::ranges::sort(gates);
	if (gates.back() == UINT32_MAX) return;
	gates.erase(std::unique(gates.begin(), gates.end()), gates.end());
	Xbyak::Label active;
	for (size_t i = 0; i < gates.size(); ++i) {
		c.cmp(c.byte[At(ActiveOffset(m_plan) + gates[i])], 0);
		if (i + 1u < gates.size()) {
			c.jne(active, Near);
		} else {
			c.je(skip, Near);
		}
	}
	c.L(active);
}

void WalkerCompiler::Zero(uint32_t offset, uint32_t size) {
	auto& c = m_code;
	if (size == 0) return;
	c.mov(c.r11, c.rdi);
	c.lea(c.rdi, c.ptr[At(offset)]);
	c.mov(c.ecx, size);
	c.xor_(c.eax, c.eax);
	c.rep();
	c.stosb();
	c.mov(c.rdi, c.r11);
}

// Jumps to unknown unless the operand has a value at run time.
void WalkerCompiler::Branch(const Operand& operand, const Xbyak::Label& unknown) {
	auto& c = m_code;
	if (operand.kind == Kind::Unknown) {
		c.jmp(unknown, Near);
	} else if (operand.kind == Kind::Node) {
		auto& node = m_nodes[operand.value];
		if (m_body && node.called) {
			auto& stub   = m_stubs.emplace_back();
			stub.operand = operand;
			stub.unknown = &unknown;
			c.cmp(c.byte[At(node.state)], 2);
			c.jne(stub.entry, Near);
			c.L(stub.back);
			return;
		}
		node.called |= m_body;
		c.call(node.label);
		c.cmp(c.byte[At(node.state)], 2);
		c.jne(unknown, Near);
	}
}

void WalkerCompiler::Load(const Xbyak::Reg64& reg, const Operand& operand) {
	auto& c = m_code;
	switch (operand.kind) {
		case Kind::Constant: c.mov(reg, operand.value); break;
		case Kind::UserData:
			m_user_data = std::max(m_user_data, static_cast<uint32_t>(operand.value) + 1u);
			c.mov(reg.cvt32(), c.dword[c.rdi + operand.value * sizeof(uint32_t)]);
			break;
		case Kind::ShaderBase: c.mov(reg, c.qword[At(m_shader_base)]); break;
		case Kind::Node: c.mov(reg, c.qword[At(m_nodes[operand.value].slot)]); break;
		case Kind::Unknown: break;
	}
}

// rax = the 48-bit base (high << 32 | low) with its low two bits clear.
void WalkerCompiler::Base(const Operand& low, const Operand& high) {
	auto& c = m_code;
	Load(c.rax, high);
	c.shl(c.rax, 48);
	c.shr(c.rax, 16);
	Load(c.rcx, low);
	c.and_(c.ecx, ~3u);
	c.or_(c.rax, c.rcx);
}

// Loads the operand into rax, or jumps to unknown when it has no value at run time.
void WalkerCompiler::Fetch(const Operand& operand, const Xbyak::Label& unknown) {
	Branch(operand, unknown);
	Load(m_code.rax, operand);
}

// Sets eax to 1 when the condition holds, else 0.
void WalkerCompiler::Set(void (Xbyak::CodeGenerator::*condition)(const Xbyak::Operand&)) {
	(m_code.*condition)(m_code.al);
	m_code.movzx(m_code.eax, m_code.al);
}

void WalkerCompiler::Store(const Node& node) {
	auto& c = m_code;
	c.mov(c.qword[At(node.slot)], c.rax);
	c.mov(c.byte[At(node.state)], 2);
	c.ret();
}

void WalkerCompiler::EmitNode(uint32_t index) {
	auto&       c     = m_code;
	auto&       node  = m_nodes[index];
	const auto& args  = node.args;
	const auto  state = c.byte[At(node.state)];
	c.L(node.label);
	c.cmp(state, 0);
	c.jne(m_ret, Near);
	c.mov(state, 1);
	if (node.result != Operand {Kind::Node, index}) {
		// A cycle referenced this value before it folded to another operand.
		Fetch(node.result, m_ret);
		Store(node);
		return;
	}
	switch (node.inst->GetOpcode()) {
		case ValueOpcode::Phi: {
			// Only predecessors reached by this walk contribute, and they must agree. The state is 3
			// once one of them has stored the value; callers read it like 1, as unknown.
			const auto& blocks = m_plan.control_phi_predecessors.at(node.inst);
			for (size_t i = 0; i < args.size(); ++i) {
				Xbyak::Label next, first;
				c.cmp(c.byte[At(m_reach + blocks[i])], 0);
				c.je(next, Near);
				Fetch(args[i], m_ret);
				c.cmp(state, 1);
				c.je(first, Near);
				c.cmp(c.rax, c.qword[At(node.slot)]);
				c.jne(m_ret, Near);
				c.jmp(next, Near);
				c.L(first);
				c.mov(c.qword[At(node.slot)], c.rax);
				c.mov(state, 3);
				c.L(next);
			}
			c.cmp(state, 3);
			c.jne(m_ret, Near);
			c.mov(state, 2);
			c.ret();
			return;
		}
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::CompositeExtractU32x3:
		case ValueOpcode::CompositeExtractU32x4: EmitRead(node); return;
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32: {
			Xbyak::Label other;
			Fetch(args[0], m_ret);
			c.test(c.rax, c.rax);
			c.jz(other, Near);
			Fetch(args[1], m_ret);
			Store(node);
			c.L(other);
			Fetch(args[2], m_ret);
			Store(node);
			return;
		}
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::UGreaterThanEqual32: {
			const bool less = node.inst->GetOpcode() == ValueOpcode::ULessThan32;
			Xbyak::Label bound_zero;
			Fetch(args[0], m_ret);
			c.test(c.eax, c.eax);
			c.jz(bound_zero, Near);
			Branch(args[1], m_ret);
			Load(c.rax, args[0]);
			Load(c.rcx, args[1]);
			// ULT tests other < bound; ULE (bound <= other) and UGE (other >= bound) share setae.
			c.cmp(c.ecx, c.eax);
			Set(less ? &Xbyak::CodeGenerator::setb : &Xbyak::CodeGenerator::setae);
			Store(node);
			c.L(bound_zero);
			c.mov(c.eax, uint32_t {!less});
			Store(node);
			return;
		}
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd: {
			const bool conjunction = node.inst->GetOpcode() == ValueOpcode::LogicalAnd;
			Xbyak::Label unknown_left, refine, decided;
			const auto decide = [&] {
				if (conjunction) {
					c.jz(decided, Near);
				} else {
					c.jnz(decided, Near);
				}
			};
			Branch(args[0], unknown_left);
			if (args[0].kind != Kind::Unknown) {
				Load(c.rax, args[0]);
				c.test(c.rax, c.rax);
				decide();
				Fetch(args[1], m_ret);
				c.test(c.rax, c.rax);
				Set(&Xbyak::CodeGenerator::setne);
				Store(node);
			}
			c.L(unknown_left);
			if (MayBeUnknown(args[0])) {
				// Without the left side only a deciding right side, or a conjunction's zero
				// refined right side, is known.
				if (args[1].kind != Kind::Unknown) {
					Fetch(args[1], refine);
					c.test(c.rax, c.rax);
					decide();
				}
				c.L(refine);
				Fetch(args[2], m_ret);
				c.test(c.rax, c.rax);
				c.jnz(m_ret, Near);
			}
			c.L(decided);
			c.mov(c.eax, uint32_t {!conjunction});
			Store(node);
			return;
		}
		default: EmitStrict(node); return;
	}
}

// Raw scalar and uniform DWORD buffer loads; out-of-bounds reads are zero.
void WalkerCompiler::EmitRead(const Node& node) {
	auto&       c         = m_code;
	const Inst* read_inst = nullptr;
	uint32_t    component = 0u;
	EXIT_IF(!UnpackReadNode(*node.inst, read_inst, component));
	const auto  op     = read_inst->GetOpcode();
	const bool  vector = VectorBufferReadDwords(op) != 0u;
	auto        args   = std::span(node.args);
	Xbyak::Label zero;
	if (vector) {
		Fetch(args[0], m_ret);
		c.test(c.rax, c.rax);
		c.jz(zero, Near);
		args = args.subspan(1);
		if (args.empty()) c.jmp(m_ret, Near);
	}
	if (!args.empty()) {
		for (const auto& arg: args) Branch(arg, m_ret);
		const auto mem_offset = m_plan.memory_info[read_inst->Flags<MemoryFlags>().index].offset;
		if (vector) {
			// args: [0]=word0, [1]=word1, [2]=records, [3]=word3, [4]=index, [5]=voffset, [6]=soffset
			Load(c.r11, args[1]);
			c.shr(c.r11d, 16);
			c.mov(c.r9d, c.r11d);
			c.and_(c.r11d, 0x3fffu); // stride
			Xbyak::Label unswizzled;
			c.test(c.r11d, c.r11d);
			c.jz(unswizzled, Near);
			c.test(c.r9d, 1u << 15u);
			c.jnz(m_ret, Near);
			c.L(unswizzled);

			Load(c.rcx, args[3]);
			c.test(c.ecx, 1u << 23u); // ADD_TID_ENABLE
			c.jnz(m_ret, Near);
			c.test(c.ecx, 0xc0000000u); // TYPE != 0
			c.jnz(zero, Near);
			c.test(c.ecx, 0x7fu << 12u); // data_format == 0
			c.jz(zero, Near);

			Load(c.r9, args[5]); // voffset
			if (mem_offset != 0u) c.add(c.r9d, mem_offset);
			c.mov(c.r10d, c.r9d);
			if (component != 0u) c.add(c.r10d, component * 4u);

			c.shr(c.ecx, 28); // OOB_SELECT (bits 30..31 are 0)
			Load(c.rax, args[2]); // NumRecords
			Xbyak::Label check_idx, check_mode2, check_mode3, in_bounds;
			c.cmp(c.ecx, 2);
			c.je(check_mode2, Near);
			c.ja(check_mode3, Near);
			// mode 0 or 1
			c.test(c.ecx, c.ecx);
			c.jnz(check_idx, Near);
			c.cmp(c.r10d, c.r11d);
			c.jae(zero, Near);
			c.L(check_idx);
			Load(c.rcx, args[4]);
			c.cmp(c.ecx, c.eax);
			c.jae(zero, Near);
			c.jmp(in_bounds, Near);
			c.L(check_mode2);
			c.test(c.eax, c.eax);
			c.jz(zero, Near);
			c.jmp(in_bounds, Near);
			c.L(check_mode3);
			Load(c.rcx, args[6]);
			c.cmp(c.ecx, c.eax);
			c.ja(zero, Near);
			c.sub(c.eax, c.ecx);
			c.cmp(c.eax, 4);
			c.jb(zero, Near);
			c.sub(c.eax, 4);
			c.cmp(c.r10d, c.eax);
			c.ja(zero, Near);
			c.L(in_bounds);

			Load(c.rax, args[4]);
			c.imul(c.eax, c.r11d);
			c.add(c.r9d, c.eax);
			Load(c.rcx, args[6]);
			c.add(c.r9d, c.ecx); // zero-extends 32-bit byte offset to r9

			Load(c.rax, args[1]);
			c.movzx(c.eax, c.ax);
			c.shl(c.rax, 32);
			Load(c.rcx, args[0]);
			c.mov(c.ecx, c.ecx);
			c.or_(c.rax, c.rcx);
			c.add(c.rax, c.r9);
			c.and_(c.rax, ~int32_t {3});
			if (component != 0u) {
				c.lea(c.r10, c.ptr[c.rax + component * 4u]);
			} else {
				c.mov(c.r10, c.rax);
			}
			c.mov(c.rcx, c.r10);
			c.shr(c.rcx, 48);
			c.jnz(m_ret, Near);
		} else {
			const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem_offset)) & ~int64_t {3};
			// A constant offset folds into the displacement; r10 = the address.
			const auto constant = int64_t {static_cast<uint32_t>(args[2].value) & ~3u} + immediate;
			const bool folded   = args[2].kind == Kind::Constant && constant + 4 <= INT32_MAX;
			const auto offset   = folded ? Xbyak::RegExp(static_cast<size_t>(constant))
			                             : c.r9 + static_cast<size_t>(immediate);
			if (args[0].kind == Kind::UserData && args[1].kind == Kind::UserData) {
				c.mov(c.rax, c.qword[At(m_bases.at({args[0].value, args[1].value}))]);
			} else {
				Base(args[0], args[1]);
			}
			if (!folded) {
				Load(c.r9, args[2]);
				c.and_(c.r9d, ~3u);
			}
			c.lea(c.r10, c.ptr[c.rax + offset]);
			if (op == ValueOpcode::LoadAddressU32) {
				c.mov(c.rcx, c.r10);
				c.shr(c.rcx, 48);
				c.jnz(m_ret, Near);
			} else {
				Load(c.r11, args[1]);
				c.shr(c.r11d, 16);
				c.and_(c.r11d, 0x3fffu);
				Load(c.rcx, args[3]);
				Xbyak::Label sized;
				c.test(c.r11d, c.r11d);
				c.jz(sized, Near);
				c.imul(c.rcx, c.r11);
				c.L(sized);
				c.lea(c.r11, c.ptr[offset + 4u]);
				c.cmp(c.r11, c.rcx);
				c.ja(zero, Near);
			}
		}
		// Every guest load is mov eax, [r10] (41 8b 02), which the fault handler completes.
		c.mov(c.eax, c.dword[c.r10]);
		++m_captures;
		c.mov(c.qword[c.r8], c.r10);
		c.add(c.r8, 8);
		Store(node);
	}
	// Only buffer reads have a zero path.
	if (op == ValueOpcode::LoadAddressU32) return;
	c.L(zero);
	c.xor_(c.eax, c.eax);
	Store(node);
}

void WalkerCompiler::EmitStrict(const Node& node) {
	auto&       c    = m_code;
	const auto& inst = *node.inst;
	for (const auto& arg: node.args) Branch(arg, m_ret);
	const std::array regs {c.rax, c.rcx, c.r9, c.r11};
	for (size_t i = 0; i < node.args.size() && i < regs.size(); ++i) Load(regs[i], node.args[i]);
	// Field ops require offset + width <= 32. Leaves (1 << width) - 1 in r11, the offset in
	// cl and the former rcx (the offset or the inserted value) in r10.
	const auto field = [&](const Xbyak::Reg32& offset, const Xbyak::Reg32& width) {
		c.lea(c.r10, c.ptr[offset.cvt64() + width.cvt64()]);
		c.cmp(c.r10, 32);
		c.ja(m_ret, Near);
		c.mov(c.r10, c.rcx);
		c.mov(c.ecx, width);
		c.mov(c.r11d, 1);
		c.shl(c.r11, c.cl);
		c.dec(c.r11);
		c.mov(c.ecx, offset.getIdx() == Xbyak::Operand::RCX ? c.r10d : offset);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::CompositeConstructU64:
			c.shl(c.rcx, 32);
			c.or_(c.rax, c.rcx);
			break;
		case ValueOpcode::CompositeExtractU64:
			if (inst.Arg(1).Resolve().U32() != 0u) c.shr(c.rax, 32);
			c.mov(c.eax, c.eax);
			break;
		case ValueOpcode::IAdd32: c.add(c.eax, c.ecx); break;
		case ValueOpcode::IAdd64: c.add(c.rax, c.rcx); break;
		case ValueOpcode::ISub32: c.sub(c.eax, c.ecx); break;
		case ValueOpcode::ISub64: c.sub(c.rax, c.rcx); break;
		case ValueOpcode::IMul32: c.imul(c.eax, c.ecx); break;
		case ValueOpcode::UMulHi:
			c.imul(c.rax, c.rcx);
			c.shr(c.rax, 32);
			break;
		case ValueOpcode::UMin32:
			c.cmp(c.eax, c.ecx);
			c.cmova(c.eax, c.ecx);
			break;
		case ValueOpcode::UMax32:
			c.cmp(c.eax, c.ecx);
			c.cmovb(c.eax, c.ecx);
			break;
		case ValueOpcode::SMin32:
			c.cmp(c.eax, c.ecx);
			c.cmovg(c.eax, c.ecx);
			break;
		case ValueOpcode::SMax32:
			c.cmp(c.eax, c.ecx);
			c.cmovl(c.eax, c.ecx);
			break;
		case ValueOpcode::ConvertF32U32:
			c.cvtsi2ss(c.xmm0, c.rax);
			c.movd(c.eax, c.xmm0);
			break;
		case ValueOpcode::ConvertU32F32: {
			// The conversion saturates and maps NaN to zero.
			Xbyak::Label done;
			c.movd(c.xmm0, c.eax);
			c.xorps(c.xmm1, c.xmm1);
			c.xor_(c.eax, c.eax);
			c.ucomiss(c.xmm0, c.xmm1);
			c.jbe(done, Near);
			c.mov(c.ecx, 0x4f800000u);
			c.movd(c.xmm1, c.ecx);
			c.mov(c.eax, UINT32_MAX);
			c.ucomiss(c.xmm0, c.xmm1);
			c.jae(done, Near);
			c.cvttss2si(c.rax, c.xmm0);
			c.L(done);
			break;
		}
		case ValueOpcode::ConvertS32F32: {
			// The conversion saturates and maps NaN to zero. cvttss2si's out-of-range result,
			// INT32_MIN, is already right for negative overflow.
			Xbyak::Label done;
			c.movd(c.xmm0, c.eax);
			c.xor_(c.eax, c.eax);
			c.ucomiss(c.xmm0, c.xmm0);
			c.jp(done, Near);
			c.mov(c.ecx, 0x4f000000u);
			c.movd(c.xmm1, c.ecx);
			c.mov(c.eax, INT32_MAX);
			c.ucomiss(c.xmm0, c.xmm1);
			c.jae(done, Near);
			c.cvttss2si(c.eax, c.xmm0);
			c.L(done);
			break;
		}
		case ValueOpcode::FPMul32:
			c.movd(c.xmm0, c.eax);
			c.movd(c.xmm1, c.ecx);
			c.mulss(c.xmm0, c.xmm1);
			c.movd(c.eax, c.xmm0);
			break;
		case ValueOpcode::FPTrunc32:
			c.movd(c.xmm0, c.eax);
			c.roundss(c.xmm0, c.xmm0, 3);
			c.movd(c.eax, c.xmm0);
			break;
		case ValueOpcode::FPRecipIFlag32:
			// Normal positive powers of two have exact normal reciprocals in every FP mode.
			c.test(c.eax, 0x807fffffu);
			c.jnz(m_ret, Near);
			c.shr(c.eax, 23);
			c.jz(m_ret, Near);
			c.cmp(c.eax, 254);
			c.jae(m_ret, Near);
			c.neg(c.eax);
			c.add(c.eax, 254);
			c.shl(c.eax, 23);
			break;
		case ValueOpcode::FPIsNan32:
			c.and_(c.eax, 0x7fffffffu);
			c.cmp(c.eax, 0x7f800000u);
			Set(&Xbyak::CodeGenerator::seta);
			break;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (inst.Flags<FPCompareFlags>().flush_input_denorms) {
				for (const auto& reg: {c.eax, c.ecx}) {
					Xbyak::Label normal;
					c.mov(c.r11d, reg);
					c.and_(c.r11d, 0x7fffffffu);
					c.cmp(c.r11d, 0x00800000u);
					c.jae(normal, Near);
					c.and_(reg, 0x80000000u);
					c.L(normal);
				}
			}
			c.movd(c.xmm0, c.eax);
			c.movd(c.xmm1, c.ecx);
			// Unordered operands set CF, so AE is false for NaN.
			if (inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32) {
				c.ucomiss(c.xmm1, c.xmm0);
			} else {
				c.ucomiss(c.xmm0, c.xmm1);
			}
			Set(&Xbyak::CodeGenerator::setae);
			break;
		case ValueOpcode::BitwiseAnd32: c.and_(c.eax, c.ecx); break;
		case ValueOpcode::BitwiseAnd64: c.and_(c.rax, c.rcx); break;
		case ValueOpcode::BitwiseOr32: c.or_(c.eax, c.ecx); break;
		case ValueOpcode::BitwiseXor32: c.xor_(c.eax, c.ecx); break;
		case ValueOpcode::BitwiseNot32: c.not_(c.eax); break;
		case ValueOpcode::ShiftLeftLogical32: c.shl(c.eax, c.cl); break;
		case ValueOpcode::ShiftLeftLogical64: c.shl(c.rax, c.cl); break;
		case ValueOpcode::ShiftRightLogical32: c.shr(c.eax, c.cl); break;
		case ValueOpcode::ShiftRightLogical64: c.shr(c.rax, c.cl); break;
		case ValueOpcode::ShiftRightArithmetic32: c.sar(c.eax, c.cl); break;
		case ValueOpcode::ShiftRightArithmetic64: c.sar(c.rax, c.cl); break;
		case ValueOpcode::BitFieldUExtract:
			field(c.ecx, c.r9d);
			c.shr(c.rax, c.cl);
			c.and_(c.rax, c.r11);
			break;
		case ValueOpcode::BitFieldSExtract: {
			Xbyak::Label done;
			field(c.ecx, c.r9d);
			c.shr(c.rax, c.cl);
			c.and_(c.rax, c.r11);
			// Width zero tests bit 63, which is clear.
			c.lea(c.ecx, c.ptr[c.r9 - 1]);
			c.bt(c.rax, c.rcx);
			c.jnc(done, Near);
			c.not_(c.r11);
			c.or_(c.rax, c.r11);
			c.mov(c.eax, c.eax);
			c.L(done);
			break;
		}
		case ValueOpcode::BitFieldInsert:
			field(c.r9d, c.r11d);
			c.shl(c.r11, c.cl);
			c.shl(c.r10, c.cl);
			c.and_(c.r10, c.r11);
			c.not_(c.r11);
			c.and_(c.rax, c.r11);
			c.or_(c.rax, c.r10);
			c.mov(c.eax, c.eax);
			break;
		case ValueOpcode::IEqual32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::sete);
			break;
		case ValueOpcode::INotEqual32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::setne);
			break;
		case ValueOpcode::IEqual64:
			c.cmp(c.rax, c.rcx);
			Set(&Xbyak::CodeGenerator::sete);
			break;
		case ValueOpcode::INotEqual64:
			c.cmp(c.rax, c.rcx);
			Set(&Xbyak::CodeGenerator::setne);
			break;
		case ValueOpcode::UGreaterThan32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::seta);
			break;
		case ValueOpcode::SGreaterThanEqual32:
			c.cmp(c.eax, c.ecx);
			Set(&Xbyak::CodeGenerator::setge);
			break;
		case ValueOpcode::LogicalXor:
			c.test(c.rax, c.rax);
			c.setne(c.al);
			c.test(c.rcx, c.rcx);
			c.setne(c.cl);
			c.xor_(c.al, c.cl);
			c.movzx(c.eax, c.al);
			break;
		case ValueOpcode::LogicalNot:
			c.test(c.rax, c.rax);
			Set(&Xbyak::CodeGenerator::sete);
			break;
		default: EXIT("unexpected SRT walker opcode\n");
	}
	Store(node);
}

void WalkerCompiler::Compile(std::span<const Value> values) {
	auto&       c         = m_code;
	const auto& sources   = m_plan.descriptor_sources;
	const auto& blocks    = m_plan.control_flow;
	const bool  resources = values.empty();
	std::vector<std::vector<uint32_t>> gates(sources.size());
	if (resources) Consume(gates);
	for (const auto value: values) AddOutput(value, UINT32_MAX);
	// Analyze every root first, so thunk bodies only reuse finished operands.
	std::vector<Operand>              conditions, reads, outputs;
	std::vector<std::vector<Operand>> descriptors(sources.size());
	if (resources) {
		for (const auto& block: blocks) conditions.push_back(Evaluate(block.condition, 0).operand);
		for (const auto& read: m_plan.srt_reads) reads.push_back(Evaluate(read.value, 0).operand);
		for (uint32_t source = 0; source < sources.size(); ++source) {
			if (gates[source].empty()) continue;
			for (uint32_t word = 0; word < sources[source].dword_count; ++word)
				descriptors[source].push_back(Evaluate(sources[source].dwords[word], 0).operand);
		}
	}
	for (const auto value: m_outputs) outputs.push_back(Evaluate(value, 0).operand);
	Layout();

	Xbyak::Label pass, fail, reset;
	// Unknown targets of walk body references; stubs jump to them after the body.
	std::deque<Xbyak::Label> exits;
	c.add(c.rdx, m_bias);
	c.mov(c.qword[At(m_shader_base)], c.rcx);
	for (const auto& [pair, slot]: m_bases) {
		Base({Kind::UserData, pair.first}, {Kind::UserData, pair.second});
		c.mov(c.qword[At(slot)], c.rax);
	}
	Zero(m_reach, m_states_end - m_reach);
	c.lea(c.r8, c.ptr[At(m_plan.srt_captures)]);
	if (!resources) {
		for (uint32_t block = 0; block < blocks.size(); ++block)
			c.mov(c.byte[At(m_reach + block)], 1);
	} else if (!blocks.empty()) {
		// Reachability grows monotonically. A newly reached predecessor of an evaluated
		// control Phi restarts the walk with fresh values; a retreating edge resumes at its target.
		std::vector<std::vector<uint32_t>> rechecks(blocks.size());
		for (uint32_t index = 0; index < m_nodes.size(); ++index) {
			const auto& node = m_nodes[index];
			if (node.inst->GetOpcode() != ValueOpcode::Phi || node.result != Operand {Kind::Node, index}) continue;
			for (const auto block: m_plan.control_phi_predecessors.at(node.inst)) rechecks[block].push_back(node.state);
		}
		std::vector<Xbyak::Label> starts(blocks.size());
		c.mov(c.byte[At(m_reach)], 1);
		c.L(pass);
		for (uint32_t block = 0; block < blocks.size(); ++block) {
			Xbyak::Label next;
			const auto mark = [&](uint32_t target) {
				const auto reach = c.byte[At(m_reach + target)];
				// Reaching a later block without control Phis to recheck only sets its byte.
				if (target > block && rechecks[target].empty()) {
					c.mov(reach, 1);
					return;
				}
				Xbyak::Label reached;
				c.cmp(reach, 0);
				c.jne(reached, Near);
				c.mov(reach, 1);
				for (const auto state: rechecks[target]) {
					c.cmp(c.byte[At(state)], 0);
					c.jne(reset, Near);
				}
				if (target <= block) c.jmp(starts[target], Near);
				c.L(reached);
			};
			const auto& successors = blocks[block].successors;
			const auto  condition  = conditions[block];
			c.L(starts[block]);
			c.cmp(c.byte[At(m_reach + block)], 0);
			c.je(next, Near);
			if (condition.kind == Kind::Constant) {
				mark(successors[condition.value != 0u ? 0 : 1]);
			} else {
				if (condition.kind != Kind::Unknown) {
					auto&        both = exits.emplace_back();
					Xbyak::Label other;
					Fetch(condition, both);
					c.test(c.rax, c.rax);
					c.jz(other, Near);
					mark(successors[0]);
					c.jmp(next, Near);
					c.L(other);
					mark(successors[1]);
					c.jmp(next, Near);
					c.L(both);
				}
				for (const auto successor: successors) mark(successor);
			}
			c.L(next);
		}
	}
	if (resources) {
		const auto flat = [&](uint32_t slot) {
			const auto offset = slot < reads.size() ? m_plan.srt_reads[slot].flat_offset : UINT32_MAX;
			if (offset >= reads.size()) {
				c.jmp(fail, Near);
				return;
			}
			Fetch(reads[slot], fail);
			c.mov(c.dword[c.rsi + offset * sizeof(uint32_t)], c.eax);
		};
		if (blocks.empty()) {
			for (uint32_t slot = 0; slot < reads.size(); ++slot) flat(slot);
		}
		for (uint32_t block = 0; block < blocks.size(); ++block) {
			if (blocks[block].srt_reads.empty()) continue;
			Xbyak::Label skip;
			c.cmp(c.byte[At(m_reach + block)], 0);
			c.je(skip, Near);
			for (const auto slot: blocks[block].srt_reads) flat(slot);
			c.L(skip);
		}
		// A source or image no block lists is always active.
		for (uint32_t index = 0; index < sources.size() + m_plan.info.images.size(); ++index) {
			const bool     image = index >= sources.size();
			const uint32_t id    = image ? index - static_cast<uint32_t>(sources.size()) : index;
			bool listed = false;
			for (uint32_t block = 0; block < blocks.size(); ++block) {
				const auto& list = image ? blocks[block].images : blocks[block].sources;
				if (std::ranges::find(list, id) == list.end()) continue;
				if (listed) {
					c.or_(c.al, c.byte[At(m_reach + block)]);
				} else {
					c.mov(c.al, c.byte[At(m_reach + block)]);
				}
				listed = true;
			}
			const auto active = c.byte[At(ActiveOffset(m_plan) + index)];
			if (listed) {
				c.mov(active, c.al);
			} else {
				c.mov(active, 1);
			}
		}
		for (uint32_t source = 0; source < sources.size(); ++source) {
			if (gates[source].empty()) continue;
			auto&      skip  = exits.emplace_back();
			const auto known = c.dword[At(source * DescriptorStride + 8u * sizeof(uint32_t))];
			c.mov(known, 0);
			Gate(gates[source], skip);
			for (uint32_t word = 0; word < descriptors[source].size(); ++word) {
				Fetch(descriptors[source][word], skip);
				c.mov(c.dword[At(source * DescriptorStride + word * sizeof(uint32_t))], c.eax);
			}
			c.mov(known, 1);
			c.L(skip);
		}
	}
	for (uint32_t index = 0; index < outputs.size(); ++index) {
		auto&      skip   = exits.emplace_back();
		const auto output = c.qword[At(OutputOffset(m_plan) + index * 8u)];
		c.mov(output, -1);
		Gate(m_output_gates[index], skip);
		Fetch(outputs[index], skip);
		c.mov(c.eax, c.eax);
		c.mov(output, c.rax);
		c.L(skip);
	}
	c.mov(c.rax, c.r8);
	c.L(m_ret);
	c.ret();
	c.L(fail);
	c.xor_(c.eax, c.eax);
	c.ret();
	if (resources && !blocks.empty()) {
		c.L(reset);
		Zero(m_states, m_states_end - m_states);
		c.lea(c.r8, c.ptr[At(m_plan.srt_captures)]);
		c.jmp(pass, Near);
	}
	m_body = false;
	for (uint32_t index = 0; index < m_nodes.size(); ++index) {
		if (Thunk(index)) EmitNode(index);
	}
	for (auto& stub: m_stubs) {
		c.L(stub.entry);
		Branch(stub.operand, *stub.unknown);
		c.jmp(stub.back, Near);
	}
	m_plan.srt_scratch.assign(m_plan.srt_captures / 8u + m_captures, 0u);
	m_plan.srt_user_data = m_user_data;
	c.readyRE();
	EXIT_IF(Xbyak::GetError() != 0);
}

} // namespace

void CompileSrtWalker(ResourcePlan& plan, std::span<const Value> values) {
	plan.srt_code = std::make_unique<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
	WalkerCompiler(plan, *plan.srt_code).Compile(values);
}

bool WalkSrt(const ResourcePlan& plan, const SrtRuntime& runtime, std::vector<uint32_t>& flat,
             std::vector<std::pair<uint64_t, uint64_t>>* reads) {
	if (plan.srt_code == nullptr || runtime.user_data.size() < plan.srt_user_data) return false;
	flat.assign(plan.srt_reads.size(), 0u);
	using Walker = const uint64_t* KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*, uint64_t*, uint64_t);
	t_running_walker = plan.srt_code.get();
	const auto* end = plan.srt_code->getCode<Walker>()(runtime.user_data.data(), flat.data(),
	                                                     plan.srt_scratch.data(), runtime.shader_base);
	t_running_walker = nullptr;
	if (end == nullptr) return false;
	if (reads != nullptr) {
		// Resizing first lets the copy vectorize; emplace_back reloads the vector's end per read.
		const std::span captured {plan.srt_scratch.data() + plan.srt_captures / 8u, end};
		const auto      first = reads->size();
		reads->resize(first + captured.size());
		std::ranges::transform(captured, reads->data() + first, [](uint64_t address) {
			return std::pair {address, uint64_t {sizeof(uint32_t)}};
		});
	}
	return true;
}

bool WalkedValue(const ResourcePlan& plan, Value value, uint32_t& result) {
	value = value.Resolve();
	uint64_t bits = 0;
	if (value.IsImmediate()) {
		if (!ImmediateBits(value, bits)) return false;
		result = static_cast<uint32_t>(bits);
		return true;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return false;
	EXIT_IF(inst->WalkOutput() == UINT32_MAX);
	std::memcpy(&bits, Scratch(plan) + OutputOffset(plan) + inst->WalkOutput() * 8u, sizeof(bits));
	result = static_cast<uint32_t>(bits);
	return bits <= UINT32_MAX;
}

bool WalkedDescriptor(const ResourcePlan& plan, uint32_t source, DescriptorValue& result) {
	if (source >= plan.descriptor_sources.size()) return false;
	std::array<uint32_t, 9> words {};
	std::memcpy(words.data(), Scratch(plan) + source * DescriptorStride, DescriptorStride);
	if (words[8] == 0u) return false;
	result = {};
	result.dword_count = plan.descriptor_sources[source].dword_count;
	std::copy_n(words.begin(), result.dword_count, result.dwords.begin());
	return true;
}

bool WalkedActive(const ResourcePlan& plan, uint32_t source) {
	return Scratch(plan)[ActiveOffset(plan) + source] != 0u;
}

bool WalkedImageActive(const ResourcePlan& plan, uint32_t image) {
	return Scratch(plan)[ActiveOffset(plan) + plan.descriptor_sources.size() + image] != 0u;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

bool ValidateWaveUniformValue(const ResourcePlan& program, Value value) {
	return RuntimeValidator(program, RuntimeValueType::Any, true).Run(value);
}

uint32_t SrtWalkerLoadLength(uint64_t rip) {
	if (t_running_walker == nullptr ||
	    rip - reinterpret_cast<uint64_t>(t_running_walker->getCode()) >= t_running_walker->getSize())
		return 0;
	const auto* code = reinterpret_cast<const uint8_t*>(rip);
	return code[0] == 0x41u && code[1] == 0x8bu && code[2] == 0x02u ? 3 : 0;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
