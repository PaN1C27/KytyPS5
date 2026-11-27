#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#define XXH_INLINE_ALL
#include <xxhash.h>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint32_t SamplerBorderClampMask    = (1u << 2u) | (1u << 5u) | (1u << 8u);
constexpr uint32_t SamplerDword3ReservedMask = 0x3ffff000u;

// value and every value reaching it through Phi arms, each once, in depth-first order.
std::vector<Value> PhiWeb(Value value) {
	std::vector<Value> web;
	std::vector<Value> pending {value};
	while (!pending.empty()) {
		const auto current = pending.back().Resolve();
		pending.pop_back();
		if (std::ranges::find(web, current) != web.end()) continue;
		web.push_back(current);
		const auto* phi = current.TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi) continue;
		for (size_t arm = 0; arm < phi->NumArgs(); ++arm)
			pending.push_back(phi->Arg(arm));
	}
	return web;
}

uint32_t PossibleU32Bits(Value value, const auto& normalize) {
	std::unordered_map<const Inst*, uint32_t> known;
	struct Frame {
		Value       at;
		const Inst* inst = nullptr;
		bool        post = false;
	};
	const auto read_bits = [&](Value raw) -> uint32_t {
		const auto at = normalize(raw);
		if (at.IsImmediate()) return at.GetType() == Type::U32 ? at.U32() : UINT32_MAX;
		const auto* inst = at.TryInstruction();
		if (inst == nullptr) return UINT32_MAX;
		const auto found = known.find(inst);
		return found != known.end() ? found->second : UINT32_MAX;
	};

	std::vector<Frame> stack {{.at = value}};
	while (!stack.empty()) {
		const auto frame = stack.back();
		stack.pop_back();
		if (frame.post) {
			const auto* inst = frame.inst;
			const auto  arg  = [&](uint32_t index) { return read_bits(inst->Arg(index)); };
			uint32_t    bits = UINT32_MAX;
			switch (inst->GetOpcode()) {
				case ValueOpcode::BitwiseAnd32: bits = arg(0) & arg(1); break;
				case ValueOpcode::BitwiseOr32:
				case ValueOpcode::UMax32: bits = arg(0) | arg(1); break;
				case ValueOpcode::SelectU32: bits = arg(1) | arg(2); break;
				case ValueOpcode::Phi: {
					uint32_t merged = 0u;
					for (uint32_t index = 0; index < inst->NumArgs(); ++index) merged |= arg(index);
					if (inst->NumArgs() != 0u) bits = merged;
					break;
				}
				case ValueOpcode::ShiftLeftLogical32:
				case ValueOpcode::ShiftRightLogical32: {
					const auto shift = inst->Arg(1).Resolve();
					if (shift.IsImmediate() && shift.GetType() == Type::U32) {
						bits = inst->GetOpcode() == ValueOpcode::ShiftLeftLogical32
						           ? arg(0) << (shift.U32() & 31u)
						           : arg(0) >> (shift.U32() & 31u);
					}
					break;
				}
				case ValueOpcode::BitFieldInsert: {
					const auto offset = inst->Arg(2).Resolve();
					const auto count  = inst->Arg(3).Resolve();
					if (offset.IsImmediate() && count.IsImmediate() && offset.U32() < 32u &&
					    count.U32() <= 32u - offset.U32()) {
						const auto field =
						    (count.U32() == 32u ? UINT32_MAX : (1u << count.U32()) - 1u)
						    << offset.U32();
						bits = (arg(0) & ~field) | ((arg(1) << offset.U32()) & field);
					}
					break;
				}
				default: break;
			}
			known[inst] = bits;
			continue;
		}
		const auto at = normalize(frame.at);
		if (at.IsImmediate()) continue;
		const auto* inst = at.TryInstruction();
		if (inst == nullptr) continue;
		if (!known.try_emplace(inst, UINT32_MAX).second) continue;

		const auto push_child = [&](uint32_t index) {
			const auto child = normalize(inst->Arg(index));
			if (const auto* ci = child.IsImmediate() ? nullptr : child.TryInstruction();
			    ci != nullptr && !known.contains(ci)) {
				stack.push_back({.at = child});
			}
		};
		switch (inst->GetOpcode()) {
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::UMax32:
				stack.push_back({.inst = inst, .post = true});
				push_child(1u);
				push_child(0u);
				break;
			case ValueOpcode::SelectU32:
				stack.push_back({.inst = inst, .post = true});
				push_child(2u);
				push_child(1u);
				break;
			case ValueOpcode::Phi:
				if (inst->NumArgs() != 0u) {
					stack.push_back({.inst = inst, .post = true});
					for (uint32_t index = inst->NumArgs(); index-- > 0;) push_child(index);
				}
				break;
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftRightLogical32: {
				const auto shift = inst->Arg(1).Resolve();
				if (shift.IsImmediate() && shift.GetType() == Type::U32) {
					stack.push_back({.inst = inst, .post = true});
					push_child(0u);
				}
				break;
			}
			case ValueOpcode::BitFieldInsert: {
				const auto offset = inst->Arg(2).Resolve();
				const auto count  = inst->Arg(3).Resolve();
				if (offset.IsImmediate() && count.IsImmediate() && offset.U32() < 32u &&
				    count.U32() <= 32u - offset.U32()) {
					stack.push_back({.inst = inst, .post = true});
					push_child(1u);
					push_child(0u);
				}
				break;
			}
			default: break;
		}
	}
	return read_bits(value);
}

uint32_t PossibleU32Bits(Value value) {
	return PossibleU32Bits(value, [](Value at) { return at.Resolve(); });
}

// A predicated VGPR write supplies its new value on exactly the lanes which
// execute an access under `active`. Do not discard a selection under a different mask.
Value ActiveValue(Value value, Value active) {
	value  = value.Resolve();
	active = active.Resolve();
	while (const auto* select = value.TryInstruction()) {
		if (select->GetOpcode() != ValueOpcode::SelectU32 || select->Arg(0).Resolve() != active) break;
		value = select->Arg(1).Resolve();
	}
	return value;
}

// The scalar read a global base dword comes from; FLAT adds its VGPR offset to it.
const Inst* AddressBaseRead(Value word, Value active) {
	const auto* inst = ActiveValue(word, active).TryInstruction();
	if (inst != nullptr && inst->GetOpcode() == ValueOpcode::IAdd32) {
		inst = inst->Arg(0).Resolve().TryInstruction();
	}
	const bool scalar = inst != nullptr && (inst->GetOpcode() == ValueOpcode::ReadConst ||
	                                        inst->GetOpcode() == ValueOpcode::GetUserData);
	return scalar ? inst : nullptr;
}

bool IsNoncanonicalFlatAddress(Value high, Value active) {
	high = ActiveValue(high, active);
	uint32_t lower = 0, upper = UINT32_MAX;
	if (high.IsImmediate() && high.GetType() == Type::U32) {
		lower = upper = high.U32();
	} else if (const auto* inst = high.TryInstruction();
	           inst != nullptr && inst->GetOpcode() == ValueOpcode::UMedTri32) {
		// The median lies between any two constant inputs, regardless of the third.
		uint32_t count = 0;
		for (size_t index = 0; index < inst->NumArgs(); ++index) {
			const auto value = inst->Arg(index).Resolve();
			if (!value.IsImmediate() || value.GetType() != Type::U32) continue;
			if (count++ == 0) lower = upper = value.U32();
			else { lower = std::min(lower, value.U32()); upper = std::max(upper, value.U32()); }
		}
		if (count < 2) return false;
	}
	// Leave one high DWORD on each side for the signed instruction offset/carry.
	return lower > 0x00008000u && upper < 0xffff7fffu;
}

Value CanonicalizeSampleAdjustDword3(Value value) {
	for (;;) {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::BitwiseOr32) {
			if ((PossibleU32Bits(value) & ~SamplerDword3ReservedMask) == 0)
				return Value(0u);
			return value;
		}
		const auto left           = inst->Arg(0).Resolve();
		const auto right          = inst->Arg(1).Resolve();
		const bool left_reserved  = (PossibleU32Bits(left) & ~SamplerDword3ReservedMask) == 0;
		const bool right_reserved = (PossibleU32Bits(right) & ~SamplerDword3ReservedMask) == 0;
		if (left_reserved && right_reserved) {
			return Value(0u);
		}
		if (left_reserved) {
			value = right;
		} else if (right_reserved) {
			value = left;
		} else {
			return value;
		}
	}
}

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Compute: return "compute";
		default: return "unknown";
	}
}

uint32_t ByteExtent(const MemoryInfo& memory) {
	const auto bytes = std::max((memory.data_bits + 7u) / 8u, 1u);
	const auto count = std::max(memory.data_dwords, 1u);
	const auto end   = static_cast<uint64_t>(memory.offset) + static_cast<uint64_t>(bytes) * count;
	return end > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(end);
}

// Prove a loop cannot continue after its bound fails, both on entry and after an
// arbitrary previous iteration. Header Phis are substituted simultaneously; all
// unsupported expressions remain unconstrained. This state exists only while tracking.
class LoopBoundProof {
public:
	LoopBoundProof(const Program& program, const Inst& induction, const Inst& bound)
	    : m_program(program), m_induction(induction), m_bound(bound) {}

	bool Excludes(Value condition, bool positive) {
		for (uint32_t incoming = 0; incoming < m_induction.NumArgs(); ++incoming) {
			m_incoming = m_induction.PhiBlock(incoming);
			m_current  = 1u + incoming;
			if (Evaluate(condition, true) != (positive ? 0u : 1u)) return false;
		}
		return true;
	}

	bool SelectsRange(const Inst& selected, Value active, const Inst& candidate,
	                  const Inst& carried, const Inst& mask,
	                  std::span<const std::pair<Value, bool>> exit_conditions) {
		// Inactive lanes retain a previously selected index. Initially every lane is
		// active; the bounded assignment establishes the invariant on each backedge.
		m_exclude_bound = false;
		std::vector<uint32_t> range_values;
		const auto in_range = [&] {
			const auto value = Unknown(Type::U32);
			range_values.push_back(value);
			return value;
		};
		std::unordered_map<uint32_t, uint32_t> contains_memo;
		const auto contains = [&](uint32_t root_node) -> uint32_t {
			struct Frame {
				uint32_t node = 0;
				bool     post = false;
			};
			std::vector<Frame> stack {{root_node, false}};
			while (!stack.empty()) {
				const auto frame = stack.back();
				stack.pop_back();
				if (frame.post) {
					const auto entry = m_nodes[frame.node];
					contains_memo[frame.node] =
					    Select(NodeFor(entry.variable, 0u, 1u), contains_memo.at(entry.high),
					           contains_memo.at(entry.low));
					continue;
				}
				if (contains_memo.contains(frame.node)) continue;
				if (std::ranges::find(range_values, frame.node) != range_values.end()) {
					contains_memo.emplace(frame.node, 1u);
					continue;
				}
				const auto entry = m_nodes[frame.node];
				if (entry.variable == UINT32_MAX) {
					contains_memo.emplace(frame.node, 0u);
					continue;
				}
				stack.push_back({frame.node, true});
				if (!contains_memo.contains(entry.low)) stack.push_back({entry.low, false});
				if (!contains_memo.contains(entry.high)) stack.push_back({entry.high, false});
			}
			return contains_memo.at(root_node);
		};
		const auto state = [&](bool in_body) {
			m_values[1].clear();
			const auto enabled = Unknown(Type::U1);
			for (const auto& phi: *m_induction.Parent()) {
				if (phi.GetOpcode() != ValueOpcode::Phi) continue;
				m_values[1][&phi] = EquivalentValue(m_program, Value(const_cast<Inst*>(&phi)),
				                                      Value(const_cast<Inst*>(&mask)))
				                         ? enabled : Unknown(phi.GetType());
			}
			m_values[1][&carried] = Select(enabled, Unknown(Type::U32), in_range());
			if (in_body) m_values[1][&m_bound] = 1u;
			m_values[1][&candidate] = Select(Evaluate(Value(const_cast<Inst*>(&m_bound)), true),
			                                  in_range(), Unknown(Type::U32));
		};
		for (uint32_t arm = 0; arm < carried.NumArgs(); ++arm) {
			const auto initial = mask.Arg(arm).Resolve();
			if (initial.IsImmediate() && initial.GetType() == Type::U1 && initial.U1()) continue;
			state(true);
			const auto retained = contains(Evaluate(carried.Arg(arm), true));
			if (Select(Evaluate(mask.Arg(arm), true), 1u, retained) != 1u) return false;
		}
		for (uint32_t arm = 0; arm < selected.NumArgs(); ++arm) {
			state(false);
			const auto value = Evaluate(selected.Arg(arm), true);
			m_values[1][&selected] = value;
			auto allowed = Evaluate(exit_conditions[arm].first, true);
			if (!exit_conditions[arm].second) allowed = Select(allowed, 0u, 1u);
			allowed = Select(allowed, Evaluate(active, true), 0u);
			if (Select(allowed, contains(value), 1u) != 1u) return false;
		}
		return true;
	}

private:
	struct Node {
		uint32_t variable = UINT32_MAX;
		uint32_t low = 0;
		uint32_t high = 0;
	};

	uint32_t NodeFor(uint32_t variable, uint32_t low, uint32_t high) {
		if (low == high) return low;
		const auto [it, inserted] = m_nodes_by_key.try_emplace(
		    std::array {variable, low, high}, static_cast<uint32_t>(m_nodes.size()));
		if (inserted) m_nodes.push_back({variable, low, high});
		return it->second;
	}

	uint32_t Unknown(Type type) {
		if (type == Type::U1) return NodeFor(m_variables++, 0u, 1u);
		m_nodes.emplace_back();
		return static_cast<uint32_t>(m_nodes.size() - 1u);
	}

	uint32_t Select(uint32_t condition, uint32_t yes, uint32_t no) {
		struct Frame {
			uint32_t cond = 0;
			uint32_t yes  = 0;
			uint32_t no   = 0;
			uint32_t var  = 0;
			bool     post = false;
		};
		std::vector<Frame> stack {{condition, yes, no, 0u, false}};
		const auto         trivial = [&](uint32_t c, uint32_t y, uint32_t n, uint32_t& out) {
			if (c == 0u) {
				out = n;
				return true;
			}
			if (c == 1u || y == n) {
				out = y;
				return true;
			}
			if (y == 1u && n == 0u) {
				out = c;
				return true;
			}
			if (const auto found = m_choices.find({c, y, n}); found != m_choices.end()) {
				out = found->second;
				return true;
			}
			return false;
		};
		uint32_t root_res = 0u;
		while (!stack.empty()) {
			const auto frame = stack.back();
			stack.pop_back();
			if (frame.post) {
				const auto arm = [&](uint32_t value, bool high) {
					const auto node = m_nodes[value];
					return node.variable == frame.var ? (high ? node.high : node.low) : value;
				};
				uint32_t low = 0u, high = 0u;
				if (!trivial(arm(frame.cond, false), arm(frame.yes, false), arm(frame.no, false),
				             low)) {
					low = m_choices.at(
					    {arm(frame.cond, false), arm(frame.yes, false), arm(frame.no, false)});
				}
				if (!trivial(arm(frame.cond, true), arm(frame.yes, true), arm(frame.no, true),
				             high)) {
					high = m_choices.at(
					    {arm(frame.cond, true), arm(frame.yes, true), arm(frame.no, true)});
				}
				const auto result = NodeFor(frame.var, low, high);
				m_choices.emplace(std::array {frame.cond, frame.yes, frame.no}, result);
				root_res = result;
				continue;
			}
			if (trivial(frame.cond, frame.yes, frame.no, root_res)) continue;
			const auto variable = std::min(
			    {m_nodes[frame.cond].variable, m_nodes[frame.yes].variable, m_nodes[frame.no].variable});
			const auto arm = [&](uint32_t value, bool high) {
				const auto node = m_nodes[value];
				return node.variable == variable ? (high ? node.high : node.low) : value;
			};
			stack.push_back({frame.cond, frame.yes, frame.no, variable, true});
			uint32_t dummy = 0u;
			const auto lc = arm(frame.cond, false), ly = arm(frame.yes, false),
			           ln = arm(frame.no, false);
			if (!trivial(lc, ly, ln, dummy)) stack.push_back({lc, ly, ln, 0u, false});
			const auto hc = arm(frame.cond, true), hy = arm(frame.yes, true),
			           hn = arm(frame.no, true);
			if (!trivial(hc, hy, hn, dummy)) stack.push_back({hc, hy, hn, 0u, false});
		}
		return root_res;
	}

	uint32_t Compare(const Inst& inst, uint32_t left, uint32_t right) {
		struct Frame {
			uint32_t l    = 0;
			uint32_t r    = 0;
			uint32_t var  = 0;
			bool     post = false;
		};
		const auto trivial = [&](uint32_t l, uint32_t r, uint32_t& out) {
			if (l == r) {
				if (inst.GetOpcode() == ValueOpcode::IEqual32) {
					out = 1u;
					return true;
				}
				if (inst.GetOpcode() == ValueOpcode::INotEqual32) {
					out = 0u;
					return true;
				}
			}
			const auto key = std::tuple {inst.GetOpcode(), inst.Flags<uint64_t>(), l, r};
			if (const auto found = m_predicates.find(key); found != m_predicates.end()) {
				out = found->second;
				return true;
			}
			return false;
		};
		std::vector<Frame> stack {{left, right, 0u, false}};
		uint32_t           root_res = 0u;
		while (!stack.empty()) {
			const auto frame = stack.back();
			stack.pop_back();
			if (frame.post) {
				const auto lhs = m_nodes[frame.l];
				const auto rhs = m_nodes[frame.r];
				const auto ll  = lhs.variable == frame.var ? lhs.low : frame.l;
				const auto rl  = rhs.variable == frame.var ? rhs.low : frame.r;
				const auto lh  = lhs.variable == frame.var ? lhs.high : frame.l;
				const auto rh  = rhs.variable == frame.var ? rhs.high : frame.r;
				uint32_t   low = 0u, high = 0u;
				if (!trivial(ll, rl, low)) {
					low = m_predicates.at(
					    std::tuple {inst.GetOpcode(), inst.Flags<uint64_t>(), ll, rl});
				}
				if (!trivial(lh, rh, high)) {
					high = m_predicates.at(
					    std::tuple {inst.GetOpcode(), inst.Flags<uint64_t>(), lh, rh});
				}
				const auto result = Select(NodeFor(frame.var, 0u, 1u), high, low);
				m_predicates.emplace(
				    std::tuple {inst.GetOpcode(), inst.Flags<uint64_t>(), frame.l, frame.r}, result);
				root_res = result;
				continue;
			}
			if (trivial(frame.l, frame.r, root_res)) continue;
			const auto variable = std::min(m_nodes[frame.l].variable, m_nodes[frame.r].variable);
			if (variable == UINT32_MAX) {
				const auto result = Unknown(Type::U1);
				m_predicates.emplace(
				    std::tuple {inst.GetOpcode(), inst.Flags<uint64_t>(), frame.l, frame.r}, result);
				root_res = result;
				continue;
			}
			const auto lhs = m_nodes[frame.l];
			const auto rhs = m_nodes[frame.r];
			stack.push_back({frame.l, frame.r, variable, true});
			uint32_t   dummy = 0u;
			const auto ll    = lhs.variable == variable ? lhs.low : frame.l;
			const auto rl    = rhs.variable == variable ? rhs.low : frame.r;
			if (!trivial(ll, rl, dummy)) stack.push_back({ll, rl, 0u, false});
			const auto lh = lhs.variable == variable ? lhs.high : frame.l;
			const auto rh = rhs.variable == variable ? rhs.high : frame.r;
			if (!trivial(lh, rh, dummy)) stack.push_back({lh, rh, 0u, false});
		}
		return root_res;
	}

	uint32_t Evaluate(Value root_value, bool root_current) {
		enum class Step : uint8_t {
			Enter,
			PhiLoopIncoming,
			PhiMultiArg,
			UnaryNot,
			AndLeft,
			AndRight,
			OrLeft,
			OrRight,
			SelectCond,
			SelectChosen,
			SelectTrue,
			SelectFalse,
			CompareArg0,
			CompareArg1,
		};
		struct Frame {
			Step        step      = Step::Enter;
			Value       value;
			bool        current   = false;
			const Inst* inst      = nullptr;
			uint32_t    val_a     = 0u;
			uint32_t    val_b     = 0u;
			size_t      arg_index = 0u;
			bool        has_cache = false;
		};
		std::vector<Frame> stack {{.step = Step::Enter, .value = root_value, .current = root_current}};
		uint32_t           last_res = 0u;

		const auto finish_frame = [&](Frame& frame, uint32_t result) {
			if (frame.has_cache) {
				auto& values = m_values[frame.current ? m_current : 0u];
				if (result == UINT32_MAX) {
					auto& slot = values.at(frame.inst);
					if (slot == UINT32_MAX) slot = Unknown(frame.value.GetType());
					result = slot;
				} else {
					values.at(frame.inst) = result;
				}
			}
			last_res = result;
			stack.pop_back();
		};
		const auto push_eval = [&](Value v, bool cur) {
			stack.push_back({.step = Step::Enter, .value = v, .current = cur});
		};

		while (!stack.empty()) {
			auto& frame = stack.back();
			switch (frame.step) {
				case Step::Enter: {
					frame.value = frame.value.Resolve();
					if (frame.value.IsImmediate()) {
						if (frame.value.GetType() == Type::U1) {
							finish_frame(frame, frame.value.U1() ? 1u : 0u);
							break;
						}
						bool found_lit = false;
						for (const auto& [literal, node]: m_literals) {
							if (literal == frame.value) {
								finish_frame(frame, node);
								found_lit = true;
								break;
							}
						}
						if (found_lit) break;
						const auto node = Unknown(frame.value.GetType());
						m_literals.emplace_back(frame.value, node);
						finish_frame(frame, node);
						break;
					}
					const auto* inst = frame.value.TryInstruction();
					if (frame.current && m_exclude_bound && inst == &m_bound) {
						finish_frame(frame, 0u);
						break;
					}
					auto& values = m_values[frame.current ? m_current : 0u];
					if (const auto found = values.find(inst); found != values.end()) {
						if (found->second == UINT32_MAX) {
							found->second = Unknown(frame.value.GetType());
						}
						finish_frame(frame, found->second);
						break;
					}
					values.emplace(inst, UINT32_MAX);
					frame.inst      = inst;
					frame.has_cache = true;

					switch (inst->GetOpcode()) {
						case ValueOpcode::Phi: {
							const auto invariant = ResolveInvariantPhi(m_program, frame.value);
							if (!invariant.IsEmpty()) {
								frame.step = Step::SelectChosen;
								push_eval(invariant, frame.current);
							} else if (inst->Parent() == m_induction.Parent()) {
								if (frame.current) {
									bool found_edge = false;
									for (uint32_t i = 0; i < inst->NumArgs(); ++i) {
										if (inst->PhiBlock(i) == m_incoming) {
											frame.step = Step::PhiLoopIncoming;
											push_eval(inst->Arg(i), false);
											found_edge = true;
											break;
										}
									}
									if (!found_edge) finish_frame(frame, UINT32_MAX);
								} else {
									finish_frame(frame, UINT32_MAX);
								}
							} else if (inst->NumArgs() != 0u) {
								frame.step      = Step::PhiMultiArg;
								frame.arg_index = 0u;
								push_eval(inst->Arg(0), frame.current);
							} else {
								finish_frame(frame, UINT32_MAX);
							}
							break;
						}
						case ValueOpcode::LogicalNot:
							frame.step = Step::UnaryNot;
							push_eval(inst->Arg(0), frame.current);
							break;
						case ValueOpcode::LogicalAnd:
							frame.step = Step::AndLeft;
							push_eval(inst->Arg(0), frame.current);
							break;
						case ValueOpcode::LogicalOr:
							frame.step = Step::OrLeft;
							push_eval(inst->Arg(0), frame.current);
							break;
						case ValueOpcode::SelectU1:
						case ValueOpcode::SelectU32:
							frame.step = Step::SelectCond;
							push_eval(inst->Arg(0), frame.current);
							break;
						default:
							if (inst->GetType() == Type::U1 && inst->NumArgs() == 2u &&
							    inst->Arg(0).GetType() == Type::U32 &&
							    inst->Arg(1).GetType() == Type::U32) {
								frame.step = Step::CompareArg0;
								push_eval(inst->Arg(0), frame.current);
							} else {
								finish_frame(frame, UINT32_MAX);
							}
							break;
					}
					break;
				}
				case Step::PhiLoopIncoming:
				case Step::SelectChosen:
					finish_frame(frame, last_res);
					break;
				case Step::PhiMultiArg: {
					const auto* inst = frame.inst;
					if (frame.arg_index == 0u) {
						frame.val_a = last_res;
					} else if (last_res != frame.val_a) {
						finish_frame(frame, UINT32_MAX);
						break;
					}
					if (++frame.arg_index < inst->NumArgs()) {
						push_eval(inst->Arg(frame.arg_index), frame.current);
					} else {
						finish_frame(frame, frame.val_a);
					}
					break;
				}
				case Step::UnaryNot:
					finish_frame(frame, Select(last_res, 0u, 1u));
					break;
				case Step::AndLeft: {
					if (last_res == 0u) {
						finish_frame(frame, 0u);
						break;
					}
					frame.val_a = last_res;
					frame.step  = Step::AndRight;
					push_eval(frame.inst->Arg(1), frame.current);
					break;
				}
				case Step::AndRight:
					finish_frame(frame, Select(frame.val_a, last_res, 0u));
					break;
				case Step::OrLeft: {
					if (last_res == 1u) {
						finish_frame(frame, 1u);
						break;
					}
					frame.val_a = last_res;
					frame.step  = Step::OrRight;
					push_eval(frame.inst->Arg(1), frame.current);
					break;
				}
				case Step::OrRight:
					finish_frame(frame, Select(frame.val_a, 1u, last_res));
					break;
				case Step::SelectCond: {
					const auto* inst = frame.inst;
					if (last_res == 0u) {
						frame.step = Step::SelectChosen;
						push_eval(inst->Arg(2), frame.current);
						break;
					}
					if (last_res == 1u) {
						frame.step = Step::SelectChosen;
						push_eval(inst->Arg(1), frame.current);
						break;
					}
					frame.val_a = last_res;
					frame.step  = Step::SelectTrue;
					push_eval(inst->Arg(1), frame.current);
					break;
				}
				case Step::SelectTrue:
					frame.val_b = last_res;
					frame.step  = Step::SelectFalse;
					push_eval(frame.inst->Arg(2), frame.current);
					break;
				case Step::SelectFalse:
					finish_frame(frame, Select(frame.val_a, frame.val_b, last_res));
					break;
				case Step::CompareArg0:
					frame.val_a = last_res;
					frame.step  = Step::CompareArg1;
					push_eval(frame.inst->Arg(1), frame.current);
					break;
				case Step::CompareArg1:
					finish_frame(frame, Compare(*frame.inst, frame.val_a, last_res));
					break;
			}
		}
		return last_res;
	}

	const Program& m_program;
	const Inst& m_induction;
	const Inst& m_bound;
	const Block* m_incoming = nullptr;
	uint32_t m_current = 1;
	uint32_t m_variables = 0;
	bool m_exclude_bound = true;
	std::vector<Node> m_nodes {{}, {}};
	std::vector<std::pair<Value, uint32_t>> m_literals;
	// Previous-iteration values, then current ones per incoming edge, kept across conditions.
	std::array<std::map<const Inst*, uint32_t>, 3> m_values;
	std::map<std::array<uint32_t, 3>, uint32_t> m_nodes_by_key;
	std::map<std::array<uint32_t, 3>, uint32_t> m_choices;
	std::map<std::tuple<ValueOpcode, uint64_t, uint32_t, uint32_t>, uint32_t> m_predicates;
};

class Tracker {
public:
	Tracker(Program& program, const Decoder::Program& decoded, const CFG::Graph& native_cfg,
	        ShaderStageInputInfo input)
	    : m_program(program), m_decoded(decoded), m_native_cfg(native_cfg), m_input(input),
	      m_scalar_writes(std::move(program.scalar_writes)), m_info(program.info) {
		std::ranges::sort(m_scalar_writes, {}, &Program::ScalarWrite::pc);
		m_info.buffers.clear();
		m_info.images.clear();
		m_info.samplers.clear();
		m_info.sampled_pairs.clear();
		m_info.address_bases.clear();
		m_info.user_address_bases.clear();
		m_shader_writes      = HasShaderMemoryWrites(program);
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::Write) {
					continue;
				}
				const auto index = inst.Flags<MemoryFlags>().index;
				if (index >= program.memory_info.size()) {
					m_has_address_writes = true;
					break;
				}
				const auto& memory = program.memory_info[index];
				if (memory.kind == ResourceKind::Scratch || memory.kind == ResourceKind::FlatLocal) {
					continue;
				}
				if (memory.kind == ResourceKind::Flat && memory.address_is_full &&
				    inst.NumArgs() >= 3u &&
				    IsNoncanonicalFlatAddress(inst.Arg(2), inst.Arg(inst.NumArgs() - 1u))) {
					continue;
				}
				m_has_address_writes = true;
				break;
			}
			if (m_has_address_writes) {
				break;
			}
		}
	}

	void Run() {
		if (m_program.resource_tracking_complete) {
			Fail(0, "resources already tracked");
		}
		ForwardPrivateLds();
		FoldDominatedBranchConditions(m_program.blocks);
		FoldBoundedLoopSelectors();
		PlanScalarReads();
		EliminateDeadCode(m_program.blocks);
		PlanIndirectDescriptors();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				Collect(inst);
			}
		}
		for (const auto& patch: m_handle_patches) {
			patch.handle->SetFlags<uint32_t>(patch.resource);
		}
		for (const auto& patch: m_memory_patches) {
			auto& memory    = m_program.memory_info[patch.index];
			memory.resource = patch.resource;
			if (patch.has_sampler) {
				memory.sampler = patch.sampler;
			}
		}
		std::vector<std::pair<const IndirectDescriptorPlan*, Value>> buffer_selections;
		for (const auto& plan: m_indirect_descriptors) {
			auto key = plan.key;
			if (plan.table_read != nullptr) {
				const auto* anchor = plan.table_indexed ? plan.handle : plan.table_read;
				auto* block = anchor->Parent();
				const auto where = std::ranges::find_if(block->Instructions(),
				    [&](const Inst& inst) { return &inst == anchor; });
				const auto emit = [&](ValueOpcode op, std::initializer_list<Value> args) {
					return Value(&*block->PrependNewInst(where, op, args));
				};
				const auto& read = *plan.table_read;
				const auto& memory = m_program.memory_info[read.Flags<MemoryFlags>().index];
				Value offset;
				Value valid(true);
				if (plan.table_indexed) {
					const auto* table = read.Arg(0).ResolveInstruction();
					const auto stride = emit(ValueOpcode::BitFieldUExtract,
					    {table->Arg(1), Value(16u), Value(14u)});
					const auto format = emit(ValueOpcode::BitFieldUExtract,
					    {table->Arg(3), Value(12u), Value(7u)});
					offset = emit(ValueOpcode::IAdd32,
					    {emit(ValueOpcode::IMul32, {read.Arg(1), stride}), Value(memory.offset)});
					valid = emit(ValueOpcode::LogicalAnd,
					    {emit(ValueOpcode::ULessThan32, {read.Arg(1), table->Arg(2)}),
					     emit(ValueOpcode::INotEqual32, {format, Value(0u)})});
				} else {
					const auto& indirect = *m_sources[plan.source].indirect_descriptor;
					const auto dynamic = emit(ValueOpcode::BitwiseAnd32, {read.Arg(1), Value(~3u)});
					if (m_sources[indirect.table_source].dword_count == 2u) {
						// An address table's heap starts at key 0's slot.
						offset = emit(ValueOpcode::ISub32, {dynamic, Value(indirect.table_offset)});
						valid = emit(ValueOpcode::ULessThanEqual32, {Value(indirect.table_offset), dynamic});
					} else {
						offset = emit(ValueOpcode::IAdd32, {dynamic, Value(memory.offset & ~3u)});
						valid = emit(ValueOpcode::ULessThanEqual32, {dynamic, offset});
					}
				}
				key = emit(ValueOpcode::SelectU32, {valid, offset, Value(UINT32_MAX)});
				if (!plan.table_indexed) {
					// NativeDescriptorSource proved this read reaches the image. Preserve its
					// SSA Phi edges while retaining descriptor DWORDs for other consumers.
					std::map<const Inst*, Value> projected;
					std::vector<const Inst*>     work;
					std::vector<const Inst*>     created;
					if (const auto* root_phi = plan.handle->Arg(0).Resolve().TryInstruction();
					    root_phi != nullptr && root_phi != plan.table_read &&
					    root_phi->GetOpcode() == ValueOpcode::Phi) {
						work.push_back(root_phi);
					}
					while (!work.empty()) {
						const auto* phi = work.back();
						work.pop_back();
						if (projected.contains(phi)) continue;
						auto* parent = phi->Parent();
						const auto position = std::ranges::find_if(parent->Instructions(),
						    [&](const Inst& inst) { return &inst == phi; });
						auto& selected = *parent->PrependNewInst(position, ValueOpcode::Phi);
						selected.SetFlags(Type::U32);
						projected.emplace(phi, Value(&selected));
						created.push_back(phi);
						for (uint32_t arm = 0; arm < phi->NumArgs(); ++arm) {
							const auto* child = phi->Arg(arm).Resolve().TryInstruction();
							if (child != nullptr && child != plan.table_read &&
							    child->GetOpcode() == ValueOpcode::Phi &&
							    !projected.contains(child)) {
								work.push_back(child);
							}
						}
					}
					for (const auto* phi: created) {
						auto* selected = projected.at(phi).Instruction();
						for (uint32_t arm = 0; arm < phi->NumArgs(); ++arm) {
							const auto* child = phi->Arg(arm).Resolve().TryInstruction();
							Value       mapped(UINT32_MAX);
							if (child == plan.table_read) {
								mapped = key;
							} else if (child != nullptr &&
							           child->GetOpcode() == ValueOpcode::Phi) {
								mapped = projected.at(child);
							}
							selected->AddPhiOperand(phi->PhiBlock(arm), mapped);
						}
					}
					const auto* top = plan.handle->Arg(0).Resolve().TryInstruction();
					key = top == plan.table_read ? key
					    : top != nullptr && top->GetOpcode() == ValueOpcode::Phi
					        ? projected.at(top)
					        : Value(UINT32_MAX);
				}
				plan.handle->SetArg(0, key);
				for (uint32_t word = 1; word < plan.handle->NumArgs(); ++word)
					plan.handle->SetArg(word, Value(0u));
				continue;
			}
			if (m_sources[plan.source].indirect_descriptor->table_maximum &&
			    plan.handle->GetOpcode() != ValueOpcode::GetBufferResource) {
				auto* block = plan.handle->Parent();
				const auto where = std::ranges::find_if(block->Instructions(),
				    [&](const Inst& inst) { return &inst == plan.handle; });
				key = Value(&*block->PrependNewInst(
				    where, ValueOpcode::ShiftLeftLogical32, {key, Value(2u)}));
			}
			if (plan.handle->GetOpcode() == ValueOpcode::GetBufferResource) {
				Value selected;
				for (const auto& [previous, value]: buffer_selections) {
					if (plan.selection_anchor != nullptr && previous->reads == plan.reads &&
					    previous->source == plan.source &&
					    previous->handle->Flags<uint32_t>() == plan.handle->Flags<uint32_t>() &&
					    EquivalentValue(m_program, previous->key, plan.key) &&
					    AvailableBefore(*value.Instruction(), *plan.handle)) {
						selected = value;
						break;
					}
				}
				if (selected.IsEmpty()) {
					const auto* anchor = plan.selection_anchor != nullptr ? plan.selection_anchor : plan.handle;
					auto* block = anchor->Parent();
					const auto where = std::ranges::find_if(block->Instructions(),
					    [&](const Inst& inst) { return &inst == anchor; });
					const auto& indirect = *m_sources[plan.source].indirect_descriptor;
					if (indirect.table_maximum) {
						key = Value(&*block->PrependNewInst(
						    where, ValueOpcode::ShiftLeftLogical32, {key, Value(2u)}));
					} else if (indirect.table_scalar) {
						// Wrapped affine offsets span one power-of-two congruence class.
						// Keep the existing mapping's byte key while compressing its empty gaps.
						key = plan.reads[0]->Arg(1);
						const auto shift = std::countr_zero(indirect.table_stride) - 2u;
						if (shift != 0u) key = Value(&*block->PrependNewInst(
						    where, ValueOpcode::ShiftRightLogical32, {key, Value(shift)}));
					}
					auto& selection = *block->PrependNewInst(where, ValueOpcode::GetBufferResourceIndex, {key});
					selection.SetFlags(plan.handle->Flags<uint32_t>());
					selected = Value(&selection);
					buffer_selections.emplace_back(&plan, selected);
				}
				plan.handle->SetArg(0, selected);
				for (uint32_t word = 1; word < 4u; ++word) plan.handle->SetArg(word, Value(0u));
			} else if (plan.handle->NumArgs() == 4u) {
				plan.handle->SetArg(0, key);
				for (uint32_t word = 1; word < 4u; ++word) plan.handle->SetArg(word, Value(0u));
			} else {
				if (plan.reads[0] == nullptr) plan.handle->SetArg(0, key);
				for (uint32_t dword = 0; dword < 4u; dword++) {
					plan.handle->SetArg(dword + 1u, plan.roots[dword + 4u]);
				}
				for (uint32_t dword = 5u; dword < plan.roots.size(); dword++) {
					plan.handle->SetArg(dword, plan.key);
				}
			}
			if (plan.reads[0] != nullptr && !plan.retain_reads) {
				for (uint32_t word = 0; word < plan.read_dwords; ++word)
					m_program.memory_info[plan.memory[word]].planning_only = true;
			}
		}
		// Project descriptor-only SSA onto its key. Existing Phis preserve dominance;
		// native descriptor resolution excludes their stale incoming values at each use.
		for (const auto& plan: m_indirect_descriptors) {
			if (plan.handle->GetOpcode() != ValueOpcode::GetImageResource || plan.reads[0] == nullptr)
				continue;
			for (const auto* read: std::span(plan.reads).first(plan.read_dwords)) {
				const auto first = std::ranges::find_if(m_indirect_descriptors, [&](const auto& candidate) {
					return candidate.reads[0] == read;
				});
				const auto key = first != m_indirect_descriptors.end() ? first->key : Value(0u);
				const auto uses = read->Uses();
				for (const auto& use: uses) use.user->SetArg(use.operand, key);
			}
		}
		m_program.descriptor_sources         = std::move(m_sources);
		m_program.info                       = std::move(m_info);
		m_program.resource_tracking_complete = true;
	}

private:
	struct HandlePatch {
		Inst*    handle   = nullptr;
		uint32_t resource = 0;
	};

	struct MemoryPatch {
		uint32_t index       = 0;
		uint32_t resource    = 0;
		uint32_t sampler     = 0;
		bool     has_sampler = false;
	};

	struct MemoryIndexUse {
		const Inst* owner = nullptr;
		uint32_t    count = 0;
	};

	struct ResolvedHandle {
		const Inst* handle;
		uint32_t pc;
		DescriptorSource source;
		Value planning_handle;
	};

	struct ScalarReadStep {
		Value    value;
		uint32_t use_pc;
		bool     collect; // Append the read in value: its operands have been walked.
	};

	struct IndirectDescriptorPlan {
		Inst*                      handle = nullptr;
		uint32_t                   source = 0;
		uint32_t                   read_dwords = 0;
		Value                      key;
		std::array<Value, 8>       roots {};
		std::array<uint32_t, 8>    memory {};
		std::array<const Inst*, 8> reads {};
		bool split_offsets = false;
		bool retain_reads = false;
		bool table_indexed = false;
		const Inst* table_read = nullptr;
		const Inst* selection_anchor = nullptr;
	};

	[[noreturn]] void Fail(uint32_t pc, const std::string& reason) const {
		const auto message =
		    fmt::format("shader resource tracking: hash=0x{:016x} stage={} pc=0x{:08x} {}",
		                m_program.shader_hash, StageName(m_program.stage), pc, reason);
		EXIT("%s", message.c_str());
		std::abort();
	}

	Value NativeDescriptorSource(Value value, uint32_t reg, uint32_t use_pc) const {
		value = value.Resolve();
		const auto* phi = value.TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi || m_native_cfg.blocks.empty())
			return value;
		std::vector<const Inst*> candidates;
		for (const auto node: PhiWeb(value)) {
			const auto* inst = node.TryInstruction();
			if (inst != nullptr && (inst->GetOpcode() == ValueOpcode::ReadConst ||
			                        inst->GetOpcode() == ValueOpcode::LoadAddressU32 ||
			                        inst->GetOpcode() == ValueOpcode::ReadConstBuffer ||
			                        inst->GetOpcode() == ValueOpcode::GetUserData))
				candidates.push_back(inst);
		}
		const auto source_at = [&](uint32_t pc) {
			Value source;
			bool dynamic = false;
			for (auto write = std::ranges::lower_bound(m_scalar_writes, pc, {}, &Program::ScalarWrite::pc);
			     write != m_scalar_writes.end() && write->pc == pc; ++write) {
				if (RegIndex(write->reg) != reg) continue;
				if (!write->value.IsImmediate() || write->value.GetType() != Type::U32) {
					dynamic = true;
					continue;
				}
				if (!source.IsEmpty() && source != write->value) return Value {};
				source = write->value;
			}
			if (!source.IsEmpty()) return dynamic ? Value {} : source;
			const auto native = std::ranges::lower_bound(m_decoded.instructions, pc, {},
			                                            &Decoder::Instruction::pc);
			for (const auto* candidate: candidates) {
				if (pc == UINT32_MAX) {
					if (candidate->GetOpcode() != ValueOpcode::GetUserData ||
					    RegIndex(candidate->Arg(0).ScalarRegister()) != reg)
						continue;
				} else {
					if (candidate->GetOpcode() != ValueOpcode::ReadConst &&
					    candidate->GetOpcode() != ValueOpcode::LoadAddressU32 &&
					    candidate->GetOpcode() != ValueOpcode::ReadConstBuffer) continue;
					const auto flags = candidate->Flags<MemoryFlags>();
					if (flags.pc != pc || flags.index >= m_program.memory_info.size()) continue;
					if (native == m_decoded.instructions.end() || native->pc != pc ||
					    native->dst.kind != Decoder::OperandKind::Sgpr ||
					    native->dst.reg + m_program.memory_info[flags.index].component_index != reg)
						continue;
				}
				const Value current(const_cast<Inst*>(candidate));
				if (!source.IsEmpty() && !EquivalentValue(m_program, source, current)) return Value {};
				source = current;
			}
			return source;
		};
		const auto use = std::ranges::find_if(m_native_cfg.blocks, [&](const auto& block) {
			return block.start_pc <= use_pc && use_pc < block.end_pc;
		});
		if (use == m_native_cfg.blocks.end()) return value;
		struct Position { uint32_t block; uint32_t before; };
		std::vector<Position> positions {{use->id, use_pc}};
		std::vector<bool> reached(m_native_cfg.blocks.size());
		Value selected;
		uint32_t selected_pc = UINT32_MAX;
		const auto select = [&](uint32_t pc) {
			const auto source = source_at(pc);
			if (source.IsEmpty()) return false;
			if (!selected.IsEmpty()) {
				const auto* inst = source.TryInstruction();
				if (inst != nullptr &&
				    (inst->GetOpcode() == ValueOpcode::LoadAddressU32 || inst->GetOpcode() == ValueOpcode::ReadConstBuffer) &&
				    selected_pc != pc) return false;
				if (!EquivalentValue(m_program, selected, source)) return false;
			}
			selected = source;
			selected_pc = pc;
			return true;
		};
		while (!positions.empty()) {
			const auto position = positions.back();
			positions.pop_back();
			const auto& block = m_native_cfg.blocks[position.block];
			// A backedge may revisit the use block after a write later than the original use.
			if (position.before == block.end_pc) {
				if (reached[block.id]) continue;
				reached[block.id] = true;
			}
			auto write = std::ranges::lower_bound(m_scalar_writes, position.before, {},
			                                    &Program::ScalarWrite::pc);
			bool found = false;
			while (write != m_scalar_writes.begin()) {
				--write;
				if (write->pc < block.start_pc) break;
				if (RegIndex(write->reg) != reg) continue;
				if (!select(write->pc)) return value;
				found = true;
				break;
			}
			if (found) continue;
			if (block.id == m_native_cfg.entry_block && !select(UINT32_MAX)) return value;
			for (const auto pred: block.predecessors)
				positions.push_back({pred, m_native_cfg.blocks[pred].end_pc});
		}
		return selected.IsEmpty() ? value : selected;
	}

	Value LowerDescriptorValue(Value root) {
		struct LowerFrame {
			Value                 value;
			const Inst*           inst       = nullptr;
			const Block*          branch     = nullptr;
			std::array<size_t, 2> arms {};
			bool                  post       = false;
			bool                  one_arm    = false;
		};
		const auto get_lowered = [&](Value v) -> Value {
			v                = v.Resolve();
			const auto* inst = v.TryInstruction();
			if (inst == nullptr) return v;
			const auto found = m_descriptor_values.find(inst);
			return found != m_descriptor_values.end() ? found->second : v;
		};
		std::vector<LowerFrame> stack {{.value = root}};
		while (!stack.empty()) {
			const auto frame = stack.back();
			stack.pop_back();
			if (frame.post) {
				const auto* inst   = frame.inst;
				auto        cached = m_descriptor_values.find(inst);
				if (inst->GetOpcode() != ValueOpcode::Phi) {
					Inst* projected = nullptr;
					for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
						const auto lowered = get_lowered(inst->Arg(arg));
						if (lowered == inst->Arg(arg).Resolve()) continue;
						if (projected == nullptr) {
							projected = &m_program.value_storage.emplace_back(
							    inst->GetOpcode(), inst->Flags<uint64_t>());
							for (size_t i = 0; i < inst->NumArgs(); ++i)
								projected->SetArg(i, inst->Arg(i));
						}
						projected->SetArg(arg, lowered);
					}
					if (projected != nullptr) cached->second = Value(projected);
				} else if (frame.one_arm) {
					cached->second = get_lowered(inst->Arg(frame.arms[0]));
				} else {
					const auto lowered0 = get_lowered(inst->Arg(frame.arms[0]));
					const auto lowered1 = get_lowered(inst->Arg(frame.arms[1]));
					if (!ValidateRuntimeValue(m_program, lowered0) ||
					    !ValidateRuntimeValue(m_program, lowered1)) {
						continue;
					}
					const auto& info      = *frame.branch;
					const auto& term      = info.terminator;
					const auto* incoming0 = inst->PhiBlock(frame.arms[0]);
					const auto* target0   = incoming0 == frame.branch ? inst->Parent() : incoming0;
					const auto  true_val  = term.true_block == target0 ? lowered0 : lowered1;
					const auto  false_val = term.true_block == target0 ? lowered1 : lowered0;
					auto&       selected  = m_program.value_storage.emplace_back(ValueOpcode::SelectU32);
					selected.SetArg(0, info.condition);
					selected.SetArg(1, true_val);
					selected.SetArg(2, false_val);
					cached->second = Value(&selected);
				}
				continue;
			}

			const auto  value = frame.value.Resolve();
			const auto* inst  = value.TryInstruction();
			if (inst == nullptr) continue;
			// The scalar planner owns reads and lowers their address handles separately.
			uint32_t memory_index = 0;
			if (ScalarReadMemory(*inst, memory_index) != nullptr) continue;
			if (!m_descriptor_values.emplace(inst, value).second) continue;

			if (inst->GetOpcode() != ValueOpcode::Phi) {
				stack.push_back({.inst = inst, .post = true});
				for (size_t arg = inst->NumArgs(); arg-- > 0;) {
					stack.push_back({.value = inst->Arg(arg)});
				}
				continue;
			}
			const auto* merge = inst->Parent();
			if (merge == nullptr || inst->NumPhiBlocks() != inst->NumArgs() ||
			    inst->GetType() != Type::U32) {
				continue;
			}
			std::array<size_t, 2> live_arms {};
			size_t                live_count = 0;
			bool                  bad_edge   = false;
			for (size_t arm = 0; arm < inst->NumArgs(); ++arm) {
				const auto* incoming = inst->PhiBlock(arm);
				if (incoming == nullptr) {
					bad_edge = true;
					break;
				}
				if (IsUnreachablePhiEdge(incoming, merge)) continue;
				if (live_count == 2u) {
					bad_edge = true;
					break;
				}
				live_arms[live_count++] = arm;
			}
			if (bad_edge || live_count == 0u) continue;
			if (live_count == 1u) {
				stack.push_back({.inst = inst, .arms = live_arms, .post = true, .one_arm = true});
				stack.push_back({.value = inst->Arg(live_arms[0])});
				continue;
			}
			const size_t arm0   = live_arms[0];
			const size_t arm1   = live_arms[1];
			const auto*  branch = inst->PhiBlock(arm0);
			if (branch == nullptr || inst->PhiBlock(arm1) == nullptr ||
			    branch == inst->PhiBlock(arm1)) {
				continue;
			}
			if (branch->ImmSuccessors().size() != 2u) {
				if (branch->ImmPredecessors().size() != 1u) continue;
				branch = branch->ImmPredecessors()[0];
			}
			if (branch == merge || branch->ImmSuccessors().size() != 2u) continue;
			std::array<const Block*, 2> targets {};
			bool                        valid_diamond = true;
			for (uint32_t arm = 0; arm < 2; arm++) {
				const auto* incoming = inst->PhiBlock(live_arms[arm]);
				if (incoming == merge ||
				    (incoming != branch &&
				     (incoming->ImmPredecessors().size() != 1u ||
				      incoming->ImmPredecessors()[0] != branch ||
				      incoming->ImmSuccessors().size() != 1u ||
				      incoming->ImmSuccessors()[0] != merge))) {
					valid_diamond = false;
					break;
				}
				targets[arm] = incoming == branch ? merge : incoming;
			}
			if (!valid_diamond) continue;
			const auto& info           = *branch;
			const auto& term           = info.terminator;
			const auto  predicate_type = !m_shader_writes     ? RuntimeValueType::Integer
			                           : m_has_address_writes ? RuntimeValueType::ImmutableInteger
			                                                  : RuntimeValueType::ScalarInteger;
			if (term.kind != CFG::TerminatorKind::ConditionalBranch ||
			    !((term.true_block == targets[0] && term.false_block == targets[1]) ||
			      (term.false_block == targets[0] && term.true_block == targets[1])) ||
			    !ValidateRuntimeValue(m_program, info.condition, predicate_type)) {
				continue;
			}
			stack.push_back({.inst = inst, .branch = branch, .arms = live_arms, .post = true});
			stack.push_back({.value = inst->Arg(arm1)});
			stack.push_back({.value = inst->Arg(arm0)});
		}
		return get_lowered(root);
	}

	void MakeSource(const Inst& handle, uint32_t width, bool sampler, bool sample_adjust,
	                uint32_t base_reg, DescriptorSource& descriptor, uint32_t pc) {
		const auto resolved = std::ranges::find_if(m_resolved_handles, [&](const auto& entry) {
			return entry.handle == &handle && entry.pc == pc;
		});
		if (resolved != m_resolved_handles.end()) {
			descriptor = resolved->source;
			return;
		}
		if (handle.NumArgs() != width) {
			Fail(pc, fmt::format("{} has {} descriptor dwords, expected {}",
			                     ValueOpcodeName(handle.GetOpcode()), handle.NumArgs(), width));
		}
		descriptor.dword_count = width;
		for (uint32_t i = 0; i < width; i++) {
			const auto value = base_reg != UINT32_MAX
			    ? NativeDescriptorSource(handle.Arg(i), base_reg + i, pc) : handle.Arg(i);
			descriptor.dwords[i] = LowerDescriptorValue(value);
		}
		if (sample_adjust) {
			descriptor.dwords[3] = CanonicalizeSampleAdjustDword3(descriptor.dwords[3]);
		}
		const auto dword0 = descriptor.dwords[0].Resolve();
		if (sampler && dword0.IsImmediate() && dword0.GetType() == Type::U32 &&
		    (dword0.U32() & SamplerBorderClampMask) == 0) {
			// Border color and its table index are unused unless a clamp axis selects border mode.
			descriptor.dwords[3] = Value(0u);
		}
		m_resolved_handles.push_back({&handle, pc, descriptor, {}});
	}

	uint32_t ScalarReadBase(const Inst& read) const {
		const auto flags = read.Flags<MemoryFlags>();
		if (m_program.memory_info[flags.index].kind == ResourceKind::ScalarBuffer)
			return m_program.memory_info[flags.index].resource * 4u;
		const auto native = std::ranges::lower_bound(m_decoded.instructions, flags.pc, {},
		                                            &Decoder::Instruction::pc);
		return native != m_decoded.instructions.end() && native->pc == flags.pc &&
		               native->src0.kind == Decoder::OperandKind::Sgpr
		           ? native->src0.reg : UINT32_MAX;
	}

	// Appends each scalar read after the reads its operands depend on, depth-first on an explicit
	// stack: descriptor dependencies can be thousands of values deep. Every SSA cycle passes
	// through a phi, so a value whose walk has begun is skipped.
	void CollectScalarRead(Value value, uint32_t use_pc) {
		m_srt_walk.push_back({value, use_pc, false});
		while (!m_srt_walk.empty()) {
			const auto step = m_srt_walk.back();
			m_srt_walk.pop_back();
			if (step.collect) {
				m_scalar_reads.push_back(step.value.TryInstruction());
				continue;
			}
			const auto resolved = step.value.Resolve();
			if (resolved.IsImmediate()) continue;
			auto* inst = resolved.TryInstruction();
			if (inst == nullptr) Fail(step.use_pc, "invalid typed planning value");
			if (!m_srt_visited.insert(inst).second) continue;
			uint32_t         memory_index = 0;
			const auto*      memory       = ScalarReadMemory(*inst, memory_index);
			const size_t     first_arg    = memory != nullptr ? 1u : 0u;
			uint32_t         width        = 0;
			DescriptorSource source;
			if (memory != nullptr) {
				const auto* handle = inst->Arg(0).Resolve().TryInstruction();
				width              = memory->kind == ResourceKind::ScalarBuffer ? 4u : 2u;
				if (handle == nullptr ||
				    handle->GetOpcode() != (width == 4u ? ValueOpcode::GetBufferResource
				                                        : ValueOpcode::GetAddressResource))
					Fail(step.use_pc, "scalar read has an invalid resource handle");
				MakeSource(*handle, width, false, false, ScalarReadBase(*inst), source,
				           inst->Flags<MemoryFlags>().pc);
				const auto offset = inst->Arg(1).Resolve();
				if (offset.IsImmediate() && offset.GetType() == Type::U32)
					m_srt_walk.push_back({resolved, step.use_pc, true});
			}
			for (size_t arg = inst->NumArgs(); arg-- > first_arg;)
				m_srt_walk.push_back({inst->Arg(arg), step.use_pc, false});
			for (uint32_t word = width; word-- > 0;)
				m_srt_walk.push_back({source.dwords[word], inst->Flags<MemoryFlags>().pc, false});
		}
	}

	void PlanScalarReads() {
		m_program.srt_plan_complete = false;
		m_program.srt_reads.clear();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				const auto image = ImageOpcodeInfoOf(op);
				if (BufferAccessOf(op) == BufferAccess::None &&
				    AddressOpcodeInfoOf(op).access == AddressAccess::None &&
				    image.access == ImageAccess::None) continue;
				const auto flags = inst.Flags<MemoryFlags>();
				if (flags.index >= m_program.memory_info.size())
					Fail(flags.pc, "memory metadata index is out of range");
				if (inst.NumArgs() < (image.needs_sampler ? 2u : 1u))
					Fail(flags.pc, "memory operation has no resource handle");
				const auto& memory = m_program.memory_info[flags.index];
				if ((op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarBuffer) ||
				    (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarAddress))
					Fail(flags.pc, "scalar read has incompatible scalar memory metadata");
				for (uint32_t arg = 0; arg < (image.needs_sampler ? 2u : 1u); ++arg) {
					const auto* handle = inst.Arg(arg).Resolve().TryInstruction();
					if (handle == nullptr) continue;
					const auto kind = handle->GetOpcode();
					const bool sampler = kind == ValueOpcode::GetSamplerResource;
					const uint32_t width = kind == ValueOpcode::GetImageResource ? 8u
					                     : kind == ValueOpcode::GetAddressResource ? 2u
					                     : kind == ValueOpcode::GetBufferResource || sampler ? 4u : 0u;
					if (width == 0u) continue;
					const auto base = width == 2u
					    ? (memory.kind == ResourceKind::ScalarAddress ? ScalarReadBase(inst) : UINT32_MAX)
					    : (sampler ? memory.sampler : memory.resource) * 4u;
					DescriptorSource source;
					MakeSource(*handle, width, sampler,
					           sampler && (memory.image_sample_flags & Decoder::ImageSampleFlagAdjust) != 0,
					           base, source, flags.pc);
					for (uint32_t word = 0; word < width; ++word)
						CollectScalarRead(source.dwords[word], flags.pc);
				}
			}
		}
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				uint32_t index = 0;
				if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 &&
				    ScalarReadMemory(inst, index) != nullptr && inst.Arg(1).Resolve().IsImmediate() &&
				    ValidateRuntimeValue(m_program, Value(&inst)))
					CollectScalarRead(Value(&inst), inst.Flags<MemoryFlags>().pc);
			}
		}
		for (auto* read: m_scalar_reads) {
			const auto flags = read->Flags<MemoryFlags>();
			auto& memory = m_program.memory_info[flags.index];
			const auto* handle = read->Arg(0).Resolve().TryInstruction();
			auto resolved = std::ranges::find_if(m_resolved_handles, [&](const auto& entry) {
				return entry.handle == handle && entry.pc == flags.pc;
			});
			EXIT_IF(resolved == m_resolved_handles.end());
			uint32_t bad_dword = 0;
			if (!ValidateSource(resolved->source, bad_dword)) continue;
			memory.planning_only = true;
			uint32_t slot = 0;
			for (; slot < m_program.srt_reads.size(); ++slot) {
				const auto* other = m_program.srt_reads[slot].value.Resolve().TryInstruction();
				if (other->GetOpcode() != read->GetOpcode() ||
				    m_program.memory_info[other->Flags<MemoryFlags>().index] != memory) continue;
				const auto* other_handle = other->Arg(0).Resolve().TryInstruction();
				bool same = true;
				for (uint32_t word = 0; word < resolved->source.dword_count; ++word)
					same &= EquivalentValue(m_program, resolved->source.dwords[word], other_handle->Arg(word));
				for (size_t arg = 1; arg < read->NumArgs(); ++arg)
					same &= EquivalentValue(m_program, read->Arg(arg), other->Arg(arg));
				if (same) break;
			}
			const bool keep = slot == m_program.srt_reads.size();
			if (keep) m_program.srt_reads.push_back({Value(read), slot});
			auto* block = read->Parent();
			auto& list = block->Instructions();
			const auto where = std::ranges::find_if(list, [&](const Inst& inst) {
				return &inst == read;
			});
			const auto resource = Value(&*block->PrependNewInst(where, ValueOpcode::GetSrtResource));
			const auto flat = Value(&*block->PrependNewInst(where, ValueOpcode::ReadConst,
			    {resource, Value(slot)}, read->Flags<uint64_t>()));
			const auto uses = read->Uses();
			for (const auto& use: uses) use.user->SetArg(use.operand, flat);
			for (auto& entry: m_resolved_handles) {
				for (uint32_t word = 0; word < entry.source.dword_count; ++word)
					if (entry.source.dwords[word].Resolve() == Value(read))
						entry.source.dwords[word] = flat;
			}
			for (auto* block: m_program.blocks) {
				if (block->condition.Resolve() == Value(read)) block->condition = flat;
				if (block->indirect_target.Resolve() == Value(read)) block->indirect_target = flat;
			}
			if (keep) {
				if (resolved->planning_handle.IsEmpty()) {
					bool unchanged = true;
					for (uint32_t word = 0; word < resolved->source.dword_count; ++word)
						unchanged &= handle->Arg(word).Resolve() == resolved->source.dwords[word].Resolve();
					resolved->planning_handle = read->Arg(0);
					if (!unchanged) {
						auto& retained = m_program.value_storage.emplace_back(handle->GetOpcode());
						for (uint32_t word = 0; word < resolved->source.dword_count; ++word)
							retained.SetArg(word, resolved->source.dwords[word]);
						resolved->planning_handle = Value(&retained);
					}
				}
				read->SetArg(0, resolved->planning_handle);
				read->SetParent(nullptr);
				m_program.value_storage.splice(m_program.value_storage.end(), list, where);
			} else {
				list.erase(where);
			}
		}
		m_program.srt_plan_complete = true;
	}

	bool ValidateSource(const DescriptorSource& descriptor, uint32_t& bad_dword) const {
		for (uint32_t i = 0; i < descriptor.dword_count; i++) {
			bad_dword = i;
			if (descriptor.dwords[i].Resolve().GetType() != Type::U32) {
				return false;
			}
			if (!ValidateRuntimeValue(m_program, descriptor.dwords[i])) {
				return false;
			}
		}
		return true;
	}

	uint32_t InternSource(const DescriptorSource& descriptor) {
		for (uint32_t candidate = 0; candidate < m_sources.size(); candidate++) {
			auto& current = m_sources[candidate];
			bool shared_image_heap = false;
			if (current.dword_count != descriptor.dword_count ||
			    current.indirect_descriptor.has_value() != descriptor.indirect_descriptor.has_value()) {
				continue;
			}
			if (current.indirect_descriptor.has_value()) {
				const auto& a = *current.indirect_descriptor;
				const auto& b = *descriptor.indirect_descriptor;
				// An indexed image heap already covers every DWORD start, so its domain
				// subsumes an affine scalar use of the same table and shares its snapshot.
				shared_image_heap = descriptor.dword_count == 8u &&
				    (a.table_record_bytes != 0u || b.table_record_bytes != 0u);
				if (a.selector != b.selector || a.table_maximum != b.table_maximum ||
				    a.pointer_table != b.pointer_table ||
				    a.table_source != b.table_source ||
				    (!shared_image_heap && (a.table_offset != b.table_offset ||
				                           a.table_immediate != b.table_immediate ||
				                           a.table_stride != b.table_stride)) || a.indexed != b.indexed ||
				    a.workgroup_axis != b.workgroup_axis ||
				    a.workgroup_multiplier != b.workgroup_multiplier ||
				    a.workgroup_shift != b.workgroup_shift || a.sources != b.sources ||
				    a.signed_count != b.signed_count ||
				    !EquivalentValue(m_program, a.key_count, b.key_count) ||
				    a.selector_first.IsEmpty() != b.selector_first.IsEmpty() ||
				    (!a.selector_first.IsEmpty() &&
				     !EquivalentValue(m_program, a.selector_first, b.selector_first)) ||
				    a.selector_mask.IsEmpty() != b.selector_mask.IsEmpty() ||
				    (!a.selector_mask.IsEmpty() &&
				     !EquivalentValue(m_program, a.selector_mask, b.selector_mask))) continue;
			}
			bool same = true;
			for (uint32_t i = 0; i < descriptor.dword_count; i++) {
				same = same && EquivalentValue(m_program, current.dwords[i], descriptor.dwords[i]);
			}
			if (same) {
				if (current.indirect_descriptor) {
					if (shared_image_heap) {
						current.indirect_descriptor->table_stride = 0u;
						current.indirect_descriptor->table_offset = 0u;
						current.indirect_descriptor->table_immediate = 0u;
					}
					current.indirect_descriptor->table_scalar |= descriptor.indirect_descriptor->table_scalar;
					current.indirect_descriptor->table_record_bytes = std::max(
					    current.indirect_descriptor->table_record_bytes,
					    descriptor.indirect_descriptor->table_record_bytes);
				}
				return candidate;
			}
		}
		m_sources.push_back(descriptor);
		return static_cast<uint32_t>(m_sources.size() - 1);
	}

	static bool ImmediateU32(Value value, uint32_t& result) {
		value = value.Resolve();
		if (!value.IsImmediate() || value.GetType() != Type::U32) {
			return false;
		}
		result = value.U32();
		return true;
	}

	static bool UsesOnlyImageDescriptors(const Inst& value) {
		if (value.Uses().empty()) return false;
		std::vector<const Inst*> values {&value};
		for (size_t index = 0; index < values.size(); ++index) {
			for (const auto& use: values[index]->Uses()) {
				if (use.user->GetOpcode() == ValueOpcode::GetImageResource) continue;
				if (use.user->GetOpcode() != ValueOpcode::Phi) return false;
				if (std::ranges::find(values, use.user) == values.end()) values.push_back(use.user);
			}
		}
		return true;
	}

	const MemoryInfo* ScalarReadMemory(const Inst& read, uint32_t& index) const {
		const bool address = read.GetOpcode() == ValueOpcode::LoadAddressU32;
		if (!(address ? read.NumArgs() == 4u
		              : read.GetOpcode() == ValueOpcode::ReadConstBuffer && read.NumArgs() == 2u)) {
			return nullptr;
		}
		if (address) {
			const auto high = read.Arg(2).Resolve();
			const auto enabled = read.Arg(3).Resolve();
			if (!high.IsImmediate() || high.GetType() != Type::U32 || high.U32() != 0u ||
			    !enabled.IsImmediate() || enabled.GetType() != Type::U1 || !enabled.U1()) {
				return nullptr;
			}
		}
		index = read.Flags<MemoryFlags>().index;
		if (index >= m_program.memory_info.size()) {
			return nullptr;
		}
		const auto& memory = m_program.memory_info[index];
		return memory.kind == (address ? ResourceKind::ScalarAddress : ResourceKind::ScalarBuffer) &&
		               memory.data_bits == 32u && memory.data_dwords == 1u
		           ? &memory
		           : nullptr;
	}

	bool MemoryIndexBelongsTo(uint32_t index, const Inst& owner) {
		if (m_memory_index_uses.empty()) {
			// Only descriptor planning asks, and it adds no memory instructions.
			m_memory_index_uses.resize(m_program.memory_info.size());
			for (const auto* block: m_program.blocks) {
				for (const auto& inst: *block) {
					const auto op = inst.GetOpcode();
					if (BufferAccessOf(op) != BufferAccess::None ||
					    AddressOpcodeInfoOf(op).access != AddressAccess::None ||
					    ImageOpcodeInfoOf(op).access != ImageAccess::None) {
						auto& use = m_memory_index_uses[inst.Flags<MemoryFlags>().index];
						use.owner = &inst;
						use.count++;
					}
				}
			}
		}
		const auto& use = m_memory_index_uses[index];
		return use.count == 0 || (use.count == 1 && use.owner == &owner);
	}

	bool MakeRuntimeTableSource(const Inst& read, DescriptorSource& descriptor) {
		const auto* handle = read.Arg(0).Resolve().TryInstruction();
		if (handle == nullptr) return false;
		const auto width = handle->GetOpcode() == ValueOpcode::GetBufferResource ? 4u
		                 : handle->GetOpcode() == ValueOpcode::GetAddressResource ? 2u : 0u;
		if (width == 0u) {
			return false;
		}
		const auto flags = read.Flags<MemoryFlags>();
		const auto kind = m_program.memory_info[flags.index].kind;
		const auto base = kind == ResourceKind::ScalarAddress || kind == ResourceKind::ScalarBuffer
		    ? ScalarReadBase(read) : UINT32_MAX;
		MakeSource(*handle, width, false, false, base, descriptor, flags.pc);
		uint32_t bad_dword = 0;
		return ValidateSource(descriptor, bad_dword);
	}

	enum class LaneQuantifier { Any, All };
	struct EdgePredicate {
		Value condition;
		bool positive;
		LaneQuantifier lanes = LaneQuantifier::All;
	};

	// The immediate that edge proves value differs from on every lane, if any.
	std::optional<uint32_t> ExcludedBy(const EdgePredicate& edge, Value value) const {
		if (edge.lanes != LaneQuantifier::All) return {};
		const auto* test = edge.condition.TryInstruction();
		if (test != nullptr && test->NumArgs() == 2u &&
		    ((test->GetOpcode() == ValueOpcode::INotEqual32 && edge.positive) ||
		     (test->GetOpcode() == ValueOpcode::IEqual32 && !edge.positive))) {
			for (uint32_t arg = 0; arg < 2u; ++arg) {
				uint32_t immediate;
				if (ImmediateU32(test->Arg(arg), immediate) &&
				    EquivalentValue(m_program, test->Arg(arg ^ 1u), value)) return immediate;
			}
		}
		return {};
	}

	bool NonzeroOnEntry(Value value, const Block* block) const {
		if (block == nullptr) return false;
		bool witnessed = false;
		const bool guarded = GuardedOnEntry(block, [](const Block* current) {
			return current->ImmPredecessors().size() != 1u;
		}, [&](const EdgePredicate& edge) {
			return ExcludedBy(edge, value) == 0u && (witnessed = true);
		});
		return guarded && witnessed;
	}

	// The key that every path from its definition to block rules out, if any.
	std::optional<uint32_t> ExcludedKeyOnEntry(Value key, const Block* block) const {
		const auto* definition = key.Resolve().TryInstruction();
		if (definition == nullptr) return {};
		std::optional<uint32_t> excluded;
		const auto defined = [&](const Block* current) { return current == definition->Parent(); };
		const bool guarded = GuardedOnEntry(block, defined, [&](const EdgePredicate& edge) {
			const auto immediate = ExcludedBy(edge, key);
			if (!immediate || (excluded && excluded != immediate)) return false;
			excluded = immediate;
			return true;
		});
		return guarded ? excluded : std::nullopt;
	}

	// The workgroup ID axis of key; with first, also of a runtime first + workgroup ID sum.
	uint32_t WorkgroupAxis(Value key, Value* first = nullptr) const {
		const auto* builtin = key.Resolve().TryInstruction();
		uint32_t axis = UINT32_MAX;
		if (first != nullptr && builtin != nullptr && builtin->GetOpcode() == ValueOpcode::IAdd32) {
			for (uint32_t arm = 0; arm < 2u; ++arm) {
				axis = WorkgroupAxis(builtin->Arg(arm));
				const auto base = builtin->Arg(arm ^ 1u);
				if (axis == UINT32_MAX ||
				    !ValidateRuntimeValue(m_program, base, RuntimeValueType::Integer)) continue;
				*first = base;
				return axis;
			}
			return UINT32_MAX;
		}
		if (m_program.stage != ShaderType::Compute || builtin == nullptr ||
		    builtin->GetOpcode() != ValueOpcode::GetBuiltin || builtin->NumArgs() != 2u ||
		    builtin->Arg(0) != Value(static_cast<uint32_t>(StageInputKind::WorkgroupId)) ||
		    !ImmediateU32(builtin->Arg(1), axis) || axis >= 3u) return UINT32_MAX;
		return axis;
	}

	// Matches key = (first + workgroup ID) * multiplier >> shift with shift >= 32; first and the
	// division are optional.
	bool MatchWorkgroupKey(Value key, DescriptorSource::IndirectDescriptor& indirect) const {
		uint32_t bits = 0, multiplier = 0;
		const auto* shift = key.Resolve().TryInstruction();
		if (shift != nullptr && shift->GetOpcode() == ValueOpcode::ShiftRightLogical32) {
			const auto* high = shift->Arg(0).Resolve().TryInstruction();
			if (high != nullptr && high->GetOpcode() == ValueOpcode::UMulHi &&
			    ImmediateU32(shift->Arg(1), bits) && bits < 32u &&
			    ImmediateU32(high->Arg(1), multiplier)) {
				key = high->Arg(0);
				bits += 32u;
			}
		}
		indirect.workgroup_axis = WorkgroupAxis(key, &indirect.selector_first);
		if (indirect.workgroup_axis == UINT32_MAX) return false;
		indirect.workgroup_multiplier = multiplier;
		indirect.workgroup_shift = bits;
		return true;
	}

	bool MatchTableOffset(Value value, Value& key, uint32_t& offset, uint32_t& stride) const {
		offset = 0;
		for (;;) {
			const auto* inst = value.Resolve().TryInstruction();
			if (inst == nullptr || inst->NumArgs() != 2u) {
				return false;
			}
			uint32_t immediate;
			if (inst->GetOpcode() == ValueOpcode::ShiftLeftLogical32 &&
			    ImmediateU32(inst->Arg(1), immediate) && immediate < 32u) {
				key = inst->Arg(0).Resolve();
				stride = 1u << immediate;
				return key.GetType() == Type::U32;
			}
			if (inst->GetOpcode() == ValueOpcode::IMul32) {
				if (ImmediateU32(inst->Arg(0), stride)) key = inst->Arg(1).Resolve();
				else if (ImmediateU32(inst->Arg(1), stride)) key = inst->Arg(0).Resolve();
				else return false;
				return stride != 0u && key.GetType() == Type::U32;
			}
			if (inst->GetOpcode() != ValueOpcode::IAdd32) {
				return false;
			}
			if (ImmediateU32(inst->Arg(0), immediate)) {
				value = inst->Arg(1);
			} else if (ImmediateU32(inst->Arg(1), immediate)) {
				value = inst->Arg(0);
			} else {
				return false;
			}
			// These additions are shader U32 arithmetic, before the scalar memory offset.
			offset += immediate;
		}
	}

	std::optional<EdgePredicate> ConditionalEdge(const Block* from, const Block* to) const {
		const auto& term = from->terminator;
		if (term.kind != CFG::TerminatorKind::ConditionalBranch ||
		    (term.true_block == to) == (term.false_block == to))
			return {};
		EdgePredicate edge {from->condition, term.true_block == to};
		while (const auto* inst = edge.condition.Resolve().TryInstruction()) {
			if (inst->GetOpcode() == ValueOpcode::LogicalNot) {
				edge.positive = !edge.positive;
			} else if (inst->GetOpcode() == ValueOpcode::ConditionRef) {
				const auto kind = inst->Flags<CFG::BranchCondition>();
				const bool scalar = kind == CFG::BranchCondition::SccZero ||
				                    kind == CFG::BranchCondition::SccNonZero;
				const bool zero = kind == CFG::BranchCondition::ExecZero ||
				                  kind == CFG::BranchCondition::VccZero;
				const bool nonzero = kind == CFG::BranchCondition::ExecNonZero ||
				                     kind == CFG::BranchCondition::VccNonZero;
				if (!scalar && !zero && !nonzero) break;
				// SCC is uniform. Negating a lane reduction exchanges all and any.
				edge.lanes = scalar || (zero == edge.positive) ? LaneQuantifier::All
				                                              : LaneQuantifier::Any;
			} else {
				break;
			}
			edge.condition = inst->Arg(0);
		}
		edge.condition = edge.condition.Resolve();
		return edge;
	}

	Value PositiveLaneWitness(const Block* use) const {
		if (use == nullptr || use->ImmPredecessors().size() != 1u) return {};
		const auto edge = ConditionalEdge(use->ImmPredecessors()[0], use);
		return edge && edge->positive ? edge->condition : Value {};
	}

	bool GuardedOnEntry(const Block* block, const auto& stops, const auto& accepts) const {
		std::vector<const Block*> pending {block};
		for (size_t i = 0; i < pending.size(); ++i) {
			const auto* current = pending[i];
			if (stops(current) || current->ImmPredecessors().empty()) return false;
			for (const auto* previous: current->ImmPredecessors()) {
				const auto edge = ConditionalEdge(previous, current);
				if (edge && accepts(*edge)) continue;
				if (std::ranges::find(pending, previous) == pending.end())
					pending.push_back(previous);
			}
		}
		return true;
	}

	bool HasActiveLane(Value mask, const Block* use) const {
		mask = SimplifyGuard(mask);
		if (mask.IsImmediate()) return mask.GetType() == Type::U1 && mask.U1();
		const auto incoming_is_nonempty = [&](const Block* from, const Block* to,
		                                     Value incoming, const Block* header) {
			const auto accepts = [&](const EdgePredicate& edge) {
				return edge.positive && Implies(edge.condition, incoming);
			};
			const auto edge = ConditionalEdge(from, to);
			if (edge && accepts(*edge)) return true;
			const auto* definition = incoming.Resolve().TryInstruction();
			return GuardedOnEntry(from, [&](const Block* block) {
				// A witness before this definition may belong to an earlier loop iteration.
				return block == header || (definition != nullptr && block == definition->Parent());
			}, accepts);
		};
		const auto* phi = mask.TryInstruction();
		if (phi != nullptr && phi->GetOpcode() == ValueOpcode::Phi &&
		    phi->GetType() == Type::U1 && phi->NumArgs() != 0u) {
			for (size_t arm = 0; arm < phi->NumArgs(); ++arm) {
				if (!incoming_is_nonempty(phi->PhiBlock(arm), phi->Parent(),
				                          phi->Arg(arm), phi->Parent())) return false;
			}
			return true;
		}
		return use != nullptr && use->ImmPredecessors().size() == 1u &&
		       incoming_is_nonempty(use->ImmPredecessors()[0], use, mask, nullptr);
	}

	Value SimplifyGuard(Value guard) const {
		const auto invariant = ResolveInvariantPhi(m_program, guard);
		return (invariant.IsEmpty() ? guard : invariant).Resolve();
	}

	bool Implies(Value root_guard, Value root_required, bool root_allow_phi = true) const {
		struct Frame {
			Value       guard;
			Value       required;
			bool        allow_phi    = true;
			const Inst* g_inst       = nullptr;
			const Inst* r_inst       = nullptr;
			uint8_t     kind         = 0u; // 0=Enter, 1=And, 2=Or, 3=Phi
			size_t      arm          = 0u;
			bool        has_entry    = false;
			bool        dep_on_cycle = false;
			bool        in_visiting  = false;
		};
		std::map<std::tuple<const Inst*, const Inst*, bool>, bool> memo;
		std::set<std::tuple<const Inst*, const Inst*, bool>>       visiting;
		std::vector<Frame>                                         stack;
		stack.push_back({.guard = root_guard, .required = root_required, .allow_phi = root_allow_phi});
		bool last_result = false;
		bool last_cycle  = false;

		const auto finish = [&](Frame& frame, bool ans, bool hit_cycle) {
			const bool cycle = frame.dep_on_cycle || hit_cycle;
			if (frame.in_visiting) {
				visiting.erase({frame.g_inst, frame.r_inst, frame.allow_phi});
				if (!cycle && frame.g_inst != nullptr && frame.r_inst != nullptr) {
					memo.emplace(std::tuple {frame.g_inst, frame.r_inst, frame.allow_phi}, ans);
				}
			}
			last_result = ans;
			last_cycle  = cycle;
			stack.pop_back();
		};

		while (!stack.empty()) {
			auto& frame = stack.back();
			if (frame.kind == 0u) {
				const auto guard    = SimplifyGuard(frame.guard);
				const auto required = SimplifyGuard(frame.required);
				if (required == Value(true) || EquivalentValue(m_program, guard, required)) {
					finish(frame, true, false);
					continue;
				}
				const auto* test = required.TryInstruction();
				if (test != nullptr && test->GetOpcode() == ValueOpcode::INotEqual32) {
					bool matched = false;
					for (uint32_t arg = 0; arg < 2u; ++arg) {
						if (test->Arg(arg).Resolve() == Value(0u) &&
						    ImpliesNonzero(guard, test->Arg(arg ^ 1u))) {
							matched = true;
							break;
						}
					}
					if (matched) {
						finish(frame, true, false);
						continue;
					}
				}
				const auto* inst = guard.TryInstruction();
				if (inst == nullptr) {
					finish(frame, false, false);
					continue;
				}
				frame.guard  = guard;
				frame.required = required;
				frame.g_inst = inst;
				frame.r_inst = test;
				if (test != nullptr) {
					if (const auto found = memo.find({inst, test, frame.allow_phi});
					    found != memo.end()) {
						finish(frame, found->second, false);
						continue;
					}
					if (!visiting.emplace(inst, test, frame.allow_phi).second) {
						finish(frame, false, true);
						continue;
					}
					frame.in_visiting = true;
				}
				const auto op = inst->GetOpcode();
				if (op == ValueOpcode::LogicalAnd) {
					frame.kind = 1u;
					stack.push_back({.guard     = inst->Arg(0),
					                 .required  = required,
					                 .allow_phi = frame.allow_phi});
					continue;
				}
				if (op == ValueOpcode::LogicalOr) {
					frame.kind = 2u;
					stack.push_back({.guard     = inst->Arg(0),
					                 .required  = required,
					                 .allow_phi = frame.allow_phi});
					continue;
				}
				if (!frame.allow_phi || op != ValueOpcode::Phi || inst->GetType() != Type::U1 ||
				    test == nullptr || test->Parent() == inst->Parent() ||
				    !AvailableBefore(*test, *inst) || inst->NumArgs() == 0u) {
					finish(frame, false, false);
					continue;
				}
				frame.kind      = 3u;
				frame.arm       = 0u;
				frame.has_entry = false;
				const bool carried = Dominates(inst->Parent(), inst->PhiBlock(0));
				frame.has_entry |= !carried;
				stack.push_back({.guard     = inst->Arg(0),
				                 .required  = carried ? guard : required,
				                 .allow_phi = false});
				continue;
			}
			frame.dep_on_cycle |= last_cycle;
			if (frame.kind == 1u) {
				if (last_result) {
					finish(frame, true, false);
					continue;
				}
				if (frame.arm == 0u) {
					frame.arm = 1u;
					stack.push_back({.guard     = frame.g_inst->Arg(1),
					                 .required  = frame.required,
					                 .allow_phi = frame.allow_phi});
					continue;
				}
				finish(frame, false, false);
				continue;
			}
			if (frame.kind == 2u) {
				if (!last_result) {
					finish(frame, false, false);
					continue;
				}
				if (frame.arm == 0u) {
					frame.arm = 1u;
					stack.push_back({.guard     = frame.g_inst->Arg(1),
					                 .required  = frame.required,
					                 .allow_phi = frame.allow_phi});
					continue;
				}
				finish(frame, true, false);
				continue;
			}
			// frame.kind == 3u (Phi)
			if (!last_result) {
				finish(frame, false, false);
				continue;
			}
			++frame.arm;
			if (frame.arm < frame.g_inst->NumArgs()) {
				const bool carried =
				    Dominates(frame.g_inst->Parent(), frame.g_inst->PhiBlock(frame.arm));
				frame.has_entry |= !carried;
				stack.push_back({.guard     = frame.g_inst->Arg(frame.arm),
				                 .required  = carried ? frame.guard : frame.required,
				                 .allow_phi = false});
				continue;
			}
			finish(frame, frame.has_entry, false);
		}
		return last_result;
	}

	Value EqualLocalValue(Value root_guard, const auto& matches) const {
		std::vector<Value>              pending {root_guard};
		std::unordered_set<const Inst*> visited;
		while (!pending.empty()) {
			const auto guard = SimplifyGuard(pending.back());
			pending.pop_back();
			const auto* inst = guard.TryInstruction();
			if (inst == nullptr || !visited.insert(inst).second) continue;
			if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
				pending.push_back(inst->Arg(1));
				pending.push_back(inst->Arg(0));
				continue;
			}
			if (inst->GetOpcode() != ValueOpcode::IEqual32 || inst->NumArgs() != 2u) continue;
			if (matches(inst->Arg(0))) return inst->Arg(1).Resolve();
			if (matches(inst->Arg(1))) return inst->Arg(0).Resolve();
		}
		return {};
	}

	Value EqualLocalKey(Value guard, Value key) const {
		return EqualLocalValue(guard, [&](Value at) { return EquivalentValue(m_program, at, key); });
	}

	struct AffineOffset {
		Value    index;
		uint64_t stride = 0;
		uint64_t offset = 0;
	};

	bool MatchAffineOffset(Value value, Value guard, AffineOffset& out,
	                       uint32_t depth = 0) const {
		if (depth > 16u) return false;
		value = ActiveValue(value, guard);
		uint32_t immediate = 0;
		if (ImmediateU32(value, immediate)) {
			out.offset = immediate;
			return true;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		if (inst->GetOpcode() == ValueOpcode::IAdd32 && inst->NumArgs() == 2u) {
			AffineOffset left, right;
			if (!MatchAffineOffset(inst->Arg(0), guard, left, depth + 1u) ||
			    !MatchAffineOffset(inst->Arg(1), guard, right, depth + 1u) ||
			    (!left.index.IsEmpty() && !right.index.IsEmpty() &&
			     !EquivalentValue(m_program, left.index, right.index))) return false;
			out.index = left.index.IsEmpty() ? right.index : left.index;
			out.stride = left.stride + right.stride;
			out.offset = left.offset + right.offset;
			return out.stride <= UINT32_MAX && out.offset <= UINT32_MAX;
		}
		if (inst->GetOpcode() == ValueOpcode::ShiftLeftLogical32 && inst->NumArgs() == 2u &&
		    ImmediateU32(inst->Arg(1), immediate) && immediate < 32u) {
			if (!MatchAffineOffset(inst->Arg(0), guard, out, depth + 1u)) return false;
			out.stride <<= immediate;
			out.offset <<= immediate;
			return out.stride <= UINT32_MAX && out.offset <= UINT32_MAX;
		}
		out.index = value;
		out.stride = 1u;
		return value.GetType() == Type::U32;
	}

	bool ImpliesNonzero(Value root_guard, Value value) const {
		std::vector<Value>              pending {root_guard};
		std::unordered_set<const Inst*> visited;
		while (!pending.empty()) {
			const auto guard = SimplifyGuard(pending.back());
			pending.pop_back();
			const auto* inst = guard.TryInstruction();
			if (inst == nullptr || !visited.insert(inst).second) continue;
			if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
				pending.push_back(inst->Arg(1));
				pending.push_back(inst->Arg(0));
				continue;
			}
			if (inst->NumArgs() != 2u) continue;
			if (inst->GetOpcode() == ValueOpcode::ULessThan32 &&
			    inst->Arg(0).Resolve() == Value(0u) &&
			    EquivalentValue(m_program, inst->Arg(1), value)) {
				return true;
			}
			if (inst->GetOpcode() != ValueOpcode::INotEqual32) continue;
			uint32_t zero = 1u;
			if ((ImmediateU32(inst->Arg(0), zero) && zero == 0u &&
			     EquivalentValue(m_program, inst->Arg(1), value)) ||
			    (ImmediateU32(inst->Arg(1), zero) && zero == 0u &&
			     EquivalentValue(m_program, inst->Arg(0), value))) {
				return true;
			}
		}
		return false;
	}

	bool ImpliesIndexBelow32(Value root_guard, Value index) const {
		std::vector<Value>              pending {root_guard};
		std::unordered_set<const Inst*> visited;
		while (!pending.empty()) {
			const auto guard = SimplifyGuard(pending.back());
			pending.pop_back();
			const auto* inst = guard.TryInstruction();
			if (inst == nullptr || !visited.insert(inst).second) continue;
			if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
				pending.push_back(inst->Arg(1));
				pending.push_back(inst->Arg(0));
				continue;
			}
			if (inst->NumArgs() != 2u) continue;
			uint32_t limit = 0;
			if ((inst->GetOpcode() == ValueOpcode::SLessThan32 &&
			     EquivalentValue(m_program, inst->Arg(0), index) &&
			     ImmediateU32(inst->Arg(1), limit) && limit == 32u) ||
			    (inst->GetOpcode() == ValueOpcode::SGreaterThan32 &&
			     ImmediateU32(inst->Arg(0), limit) && limit == 32u &&
			     EquivalentValue(m_program, inst->Arg(1), index))) {
				return true;
			}
		}
		return false;
	}

	bool MaskOnlyLosesBits(Value value, Value mask, bool nonzero) const {
		const auto* update = value.Resolve().TryInstruction();
		if (update == nullptr || update->NumArgs() != 2u) return false;
		if (update->GetOpcode() == ValueOpcode::BitwiseAnd32) {
			return EquivalentValue(m_program, update->Arg(0), mask) ||
			       EquivalentValue(m_program, update->Arg(1), mask);
		}
		// XOR of the first-bit mask adds bit 31 when FindILsb receives zero.
		if (!nonzero || update->GetOpcode() != ValueOpcode::BitwiseXor32) return false;
		Value bit;
		if (EquivalentValue(m_program, update->Arg(0), mask)) bit = update->Arg(1);
		else if (EquivalentValue(m_program, update->Arg(1), mask)) bit = update->Arg(0);
		else return false;
		const auto* shift = bit.Resolve().TryInstruction();
		uint32_t one = 0;
		if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
		    shift->NumArgs() != 2u || !ImmediateU32(shift->Arg(0), one) || one != 1u)
			return false;
		Value position = shift->Arg(1).Resolve();
		const auto* masked = position.TryInstruction();
		uint32_t lane_mask = 0;
		if (masked != nullptr && masked->GetOpcode() == ValueOpcode::BitwiseAnd32 &&
		    masked->NumArgs() == 2u) {
			if (ImmediateU32(masked->Arg(0), lane_mask) && lane_mask == 31u)
				position = masked->Arg(1);
			else if (ImmediateU32(masked->Arg(1), lane_mask) && lane_mask == 31u)
				position = masked->Arg(0);
		}
		const auto* first = position.Resolve().TryInstruction();
		return first != nullptr && first->GetOpcode() == ValueOpcode::FindILsb32 &&
		       first->NumArgs() == 1u && EquivalentValue(m_program, first->Arg(0), mask);
	}

	Value InitialCandidateMask(Value value, const Block* update_block = nullptr,
	                           bool nonzero = false) const {
		const auto* phi = value.Resolve().TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi ||
		    phi->NumArgs() != 2u || phi->GetType() != Type::U32) return {};
		for (uint32_t back = 0; back < 2u; ++back) {
			if ((update_block != nullptr && phi->PhiBlock(back) != update_block) ||
			    !MaskOnlyLosesBits(phi->Arg(back), value, nonzero)) continue;
			return phi->Arg(back ^ 1u).Resolve();
		}
		return {};
	}

	bool MaskEdge(const Block* from, const Block* to, Value predicate, bool positive) const {
		const auto edge = ConditionalEdge(from, to);
		// Continuation needs an active lane; an inactive exit must include every lane.
		return edge && edge->positive == positive &&
		       (positive || edge->lanes == LaneQuantifier::All) &&
		       EquivalentValue(m_program, edge->condition, predicate);
	}

	Value BoundedSetBitMask(Value index, Value guard) const {
		const auto* phi = index.Resolve().TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi ||
		    phi->NumArgs() != 3u || phi->GetType() != Type::U32 ||
		    !ImpliesIndexBelow32(guard, index)) return {};
		for (uint32_t bit_arm = 0; bit_arm < 3u; ++bit_arm) {
			const auto bit_guard = PositiveLaneWitness(phi->PhiBlock(bit_arm));
			const auto* selected = phi->Arg(bit_arm).Resolve().TryInstruction();
			if (selected == nullptr || selected->GetOpcode() != ValueOpcode::SelectU32 ||
			    selected->NumArgs() != 3u ||
			    !Implies(bit_guard, selected->Arg(0))) continue;
			const auto* first = selected->Arg(1).Resolve().TryInstruction();
			if (first == nullptr || first->GetOpcode() != ValueOpcode::FindILsb32 ||
			    first->NumArgs() != 1u || !ImpliesNonzero(bit_guard, first->Arg(0)))
				continue;
			const auto initial_mask = InitialCandidateMask(first->Arg(0), phi->PhiBlock(bit_arm), true);
			if (initial_mask.IsEmpty() ||
			    !ValidateRuntimeValue(m_program, initial_mask, RuntimeValueType::Integer)) continue;
			for (uint32_t sentinel_arm = 0; sentinel_arm < 3u; ++sentinel_arm) {
				if (sentinel_arm == bit_arm) continue;
				const auto sentinel_guard = PositiveLaneWitness(phi->PhiBlock(sentinel_arm));
				const auto* sentinel = phi->Arg(sentinel_arm).Resolve().TryInstruction();
				uint32_t bound = 0;
				if (sentinel == nullptr || sentinel->GetOpcode() != ValueOpcode::SelectU32 ||
				    sentinel->NumArgs() != 3u ||
				    !Implies(sentinel_guard, sentinel->Arg(0)) ||
				    !ImmediateU32(sentinel->Arg(1), bound) || bound != 32u) continue;
				const auto loop_active = sentinel->Arg(0).Resolve();
				const auto* active_phi = loop_active.TryInstruction();
				if (active_phi == nullptr || active_phi->GetOpcode() != ValueOpcode::Phi ||
				    active_phi->NumArgs() != 2u) continue;
				const auto other_arm = 3u - bit_arm - sentinel_arm;
				const auto* carried = phi->Arg(other_arm).Resolve().TryInstruction();
				const auto* bit_block = phi->PhiBlock(bit_arm);
				if (carried == nullptr || carried->GetOpcode() != ValueOpcode::Phi ||
				    carried->NumArgs() != 2u || carried->Parent() != active_phi->Parent() ||
				    selected->Arg(2).Resolve() != phi->Arg(sentinel_arm).Resolve() ||
				    sentinel->Arg(2).Resolve() != phi->Arg(other_arm).Resolve()) continue;
				const uint32_t back = carried->PhiBlock(0) == bit_block ? 0u : 1u;
				if (carried->PhiBlock(back) != bit_block ||
				    carried->Arg(back).Resolve() != phi->Arg(bit_arm).Resolve()) continue;
				bool invariant = false;
				for (uint32_t initial = 0; initial < 2u; ++initial) {
					invariant = active_phi->PhiBlock(initial) == carried->PhiBlock(back ^ 1u) &&
					            active_phi->PhiBlock(initial ^ 1u) == bit_block &&
					            Implies(guard, active_phi->Arg(initial)) &&
					            MaskEdge(bit_block, active_phi->Parent(),
					                     active_phi->Arg(initial ^ 1u), true);
					if (invariant) break;
				}
				if (!invariant) continue;
				if (MaskEdge(phi->PhiBlock(other_arm), phi->Parent(), loop_active, false))
					return initial_mask;
			}
		}
		return {};
	}

	const Inst* UniformizedMaterialValue(Value key, const Inst& use, Value executing = {}) const {
		Value guard;
		Value local;
		const auto* first = key.Resolve().TryInstruction();
		if (first != nullptr && first->GetOpcode() == ValueOpcode::ReadFirstLane &&
		    first->NumArgs() == 2u) {
			guard = first->Arg(1).Resolve();
			// Empty EXEC selects lane zero, which may not have loaded a material key.
			// An executing store also witnesses nonempty EXEC at the lane read.
			if (executing.IsEmpty() ? !HasActiveLane(guard, first->Parent())
			                        : !Implies(executing, guard)) return nullptr;
			local = first->Arg(0).Resolve();
		} else {
			guard = PositiveLaneWitness(use.Parent());
			if (guard.IsEmpty()) return nullptr;
			local = EqualLocalKey(guard, key);
		}
		const auto* selected = local.Resolve().TryInstruction();
		if (selected == nullptr || selected->GetOpcode() != ValueOpcode::SelectU32 ||
		    selected->NumArgs() != 3u || !Implies(guard, selected->Arg(0))) return nullptr;
		return selected;
	}

	const Inst* SelectorReadValue(Value value, Value active,
	                              DescriptorSource::IndirectDescriptor::SelectorRead& selector) const {
		const auto* read = ActiveValue(value, active).TryInstruction();
		if (read != nullptr && read->GetOpcode() == ValueOpcode::UMin32) {
			if (ImmediateU32(read->Arg(1), selector.maximum)) value = read->Arg(0);
			else if (ImmediateU32(read->Arg(0), selector.maximum)) value = read->Arg(1);
			else return nullptr;
			read = ActiveValue(value, active).TryInstruction();
		}
		if (read != nullptr && read->GetOpcode() == ValueOpcode::SMax32) {
			if (read->Arg(1).Resolve() == Value(0u)) value = read->Arg(0);
			else if (read->Arg(0).Resolve() == Value(0u)) value = read->Arg(1);
			else return nullptr;
			selector.zero_above = INT32_MAX;
			read = ActiveValue(value, active).TryInstruction();
		}
		if (read != nullptr && read->GetOpcode() == ValueOpcode::BitFieldUExtract) {
			uint32_t width;
			if (!ImmediateU32(read->Arg(1), selector.bit_offset) ||
			    !ImmediateU32(read->Arg(2), width) || selector.bit_offset >= 32u ||
			    width == 0u || width > 32u - selector.bit_offset) return nullptr;
			selector.bit_mask = width == 32u ? UINT32_MAX : (1u << width) - 1u;
			read = ActiveValue(read->Arg(0), active).TryInstruction();
		}
		return read;
	}

	const Inst* IndexedMaterialRead(const Inst* read, Value active,
	                               DescriptorSource::IndirectDescriptor::SelectorRead& selector) const {
		if (read == nullptr) return nullptr;
		uint32_t lane = 0;
		const auto op = read->GetOpcode();
		if (op == ValueOpcode::CompositeExtractU32x2 || op == ValueOpcode::CompositeExtractU32x3) {
			const bool x3 = op == ValueOpcode::CompositeExtractU32x3;
			if (!ImmediateU32(read->Arg(1), lane) || lane >= (x3 ? 3u : 2u)) return nullptr;
			read = read->Arg(0).Resolve().TryInstruction();
			if (read == nullptr ||
			    read->GetOpcode() != (x3 ? ValueOpcode::LoadBufferU32x3 : ValueOpcode::LoadBufferU32x2))
				return nullptr;
		} else if (op != ValueOpcode::LoadBufferU32) {
			return nullptr;
		}
		const auto& memory = m_program.memory_info[read->Flags<MemoryFlags>().index];
		if (memory.kind != ResourceKind::Buffer || memory.typed || memory.formatted ||
		    !memory.idxen || memory.offen || memory.offset > UINT32_MAX - lane * 4u ||
		    read->Arg(2).Resolve() != Value(0u) || read->Arg(3).Resolve() != Value(0u) ||
		    !Implies(active, read->Arg(4))) return nullptr;
		selector.offset = memory.offset + lane * 4u;
		return read;
	}

	// Predicated writes and SSA joins may retain the original material field or zero.
	// The outer unsigned check bounds every nonzero field used by this waterfall.
	const Inst* GuardedMaterialField(const Inst* select, Value active,
	                                DescriptorSource::IndirectDescriptor::SelectorRead& selector) const {
		uint32_t limit;
		if (select == nullptr || select->GetOpcode() != ValueOpcode::SelectU32 ||
		    select->Arg(2).Resolve() != Value(0u) ||
		    !ImmediateU32(UnsignedBound(select->Arg(0), select->Arg(1)), limit) || limit == 0u) return nullptr;
		const Inst* field = nullptr;
		std::vector<Value> pending {select->Arg(1)};
		for (size_t i = 0; i < pending.size(); ++i) {
			auto value = pending[i].Resolve();
			if (value == Value(0u)) continue;
			const auto* inst = value.TryInstruction();
			if (inst == nullptr || inst->Parent() == nullptr || InCycle(inst->Parent())) return nullptr;
			const bool phi = inst->GetOpcode() == ValueOpcode::Phi;
			if (phi && inst->NumArgs() == 0u) return nullptr;
			if (!phi && inst->GetOpcode() != ValueOpcode::SelectU32) {
				if (field != nullptr && field != inst) return nullptr;
				field = inst;
				continue;
			}
			const auto end = !phi && Implies(active, inst->Arg(0)) ? 2u : inst->NumArgs();
			for (size_t arm = phi ? 0u : 1u; arm < end; ++arm) {
				value = inst->Arg(arm).Resolve();
				if (std::ranges::find(pending, value) == pending.end()) pending.push_back(value);
			}
		}
		selector.zero_above = limit - 1u;
		return field;
	}

	enum class MaterialKey { Global, Indexed, Heap, Guarded };

	bool MatchUniformizedMaterialKey(Value key, const Inst& image,
	                                DescriptorSource::IndirectDescriptor& indirect,
	                                DescriptorSource& material_source, MaterialKey kind) {
		if (m_program.stage != ShaderType::Compute &&
		    (kind == MaterialKey::Indexed || kind == MaterialKey::Heap)) return false;
		const auto* selected = UniformizedMaterialValue(key, image);
		if (selected == nullptr) return false;
		const auto active = selected->Arg(0).Resolve();
		DescriptorSource::IndirectDescriptor::SelectorRead selector;
		const auto* read = kind == MaterialKey::Guarded
		    ? GuardedMaterialField(selected->Arg(1).Resolve().TryInstruction(), active, selector)
		    : SelectorReadValue(selected->Arg(1), active, selector);
		if (kind != MaterialKey::Global) {
			read = IndexedMaterialRead(read, active, selector);
			// Bounded heaps keep their full domain when the material may alias a write: written
			// buffers are checked at dispatch, storage images and address writes here. Pointer
			// tables have no such domain, so a key that a same-dispatch write changes selects
			// the root descriptor. Graphics heaps retain their full domain.
			if (read == nullptr ||
			    (kind == MaterialKey::Heap && std::ranges::any_of(m_program.blocks, [](const Block* block) {
				    return std::ranges::any_of(*block, [](const Inst& inst) {
					    const auto op = inst.GetOpcode();
					    return ImageOpcodeInfoOf(op).resource_class == ImageResourceClass::Storage ||
					           AddressOpcodeInfoOf(op).access == AddressAccess::Write;
				    });
			    })) || !MakeRuntimeTableSource(*read, material_source)) return false;
			// Wrapped U32 indices preserve the stride's power-of-two congruence class.
			AffineOffset row;
			if (MatchAffineOffset(read->Arg(1), active, row) && row.stride != 0u) {
				selector.record_step = 1u << std::countr_zero(static_cast<uint32_t>(row.stride));
				selector.first_record = static_cast<uint32_t>(row.offset) % selector.record_step;
			}
			selector.source = InternSource(material_source);
			indirect.selector = std::move(selector);
			return true;
		}
		if (read == nullptr || read->GetOpcode() != ValueOpcode::LoadAddressU32 ||
		    read->NumArgs() != 4u || !Implies(active, read->Arg(3)))
			return false;
		uint32_t high = 1u;
		if (!ImmediateU32(read->Arg(2), high) || high != 0u) return false;
		const auto memory_index = read->Flags<MemoryFlags>().index;
		if (memory_index >= m_program.memory_info.size()) return false;
		const auto& memory = m_program.memory_info[memory_index];
		if (memory.kind != ResourceKind::Global || memory.address_is_full || memory.offset > INT32_MAX ||
		    memory.data_bits != 32u ||
		    memory.data_dwords != 1u || !MemoryIndexBelongsTo(memory_index, *read)) return false;
		const auto* material_handle = read->Arg(0).Resolve().TryInstruction();
		if (material_handle == nullptr ||
		    material_handle->GetOpcode() != ValueOpcode::GetAddressResource ||
		    !MakeRuntimeTableSource(*read, material_source)) return false;
		const auto bits = PossibleU32Bits(read->Arg(1), [&](Value at) {
			at = ActiveValue(at, active);
			const auto* inst = at.TryInstruction();
			// The current lane predicate cannot constrain a prior loop iteration.
			return inst != nullptr && inst->GetOpcode() == ValueOpcode::Phi && InCycle(inst->Parent())
			           ? Value{} : at;
		});
		const auto shift = std::countr_zero(bits);
		const auto count = bits == 0u ? 0u : uint64_t {bits >> shift} + 1u;
		if (shift >= 2u && count != 0u && count <= MaxIndirectDescriptorProbes &&
		    std::has_single_bit(count) && uint64_t {bits} + memory.offset + 4u <= UINT32_MAX + 1ull) {
			selector.stride = 1u << shift;
			selector.offset = memory.offset;
			indirect.key_count = Value(static_cast<uint32_t>(count));
			indirect.selector_first = Value(0u);
		} else {
			AffineOffset offset;
			if (!MatchAffineOffset(read->Arg(1), active, offset) || offset.index.IsEmpty() ||
			    offset.stride == 0u || offset.offset + memory.offset > UINT32_MAX ||
			    offset.offset + memory.offset + 31u * offset.stride + 4u >
			        static_cast<uint64_t>(UINT32_MAX) + 1u) return false;
			const auto mask = BoundedSetBitMask(offset.index, active);
			if (mask.IsEmpty()) return false;
			selector.stride = static_cast<uint32_t>(offset.stride);
			selector.offset = static_cast<uint32_t>(offset.offset + memory.offset);
			indirect.key_count = Value(32u);
			indirect.selector_mask = mask;
		}
		selector.source = InternSource(material_source);
		selector.global = true;
		indirect.selector = std::move(selector);
		return true;
	}

	struct Induction {
		const Inst* phi;
		const Inst* step;
		uint32_t initial_arm;
	};

	static bool ZeroBased(uint32_t initial, uint32_t step) { return initial == 0u && step == 1u; }

	std::optional<Induction> MatchInduction(Value key, const auto& accepts_step) const {
		key = key.Resolve();
		const auto* phi = key.TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi || phi->GetType() != Type::U32 ||
		    phi->NumArgs() != 2u || phi->NumPhiBlocks() != 2u || phi->Parent() == nullptr)
			return {};
		for (uint32_t initial = 0; initial < 2u; ++initial) {
			const auto initial_value = phi->Arg(initial).Resolve();
			const auto* step = phi->Arg(initial ^ 1u).Resolve().TryInstruction();
			if (!initial_value.IsImmediate() || initial_value.GetType() != Type::U32 ||
			    step == nullptr || step->GetOpcode() != ValueOpcode::IAdd32 ||
			    step->Parent() == nullptr) continue;
			uint32_t increment = 0;
			if ((step->Arg(0).Resolve() == key &&
			     ImmediateU32(step->Arg(1), increment) && accepts_step(initial_value.U32(), increment)) ||
			    (step->Arg(1).Resolve() == key &&
			     ImmediateU32(step->Arg(0), increment) && accepts_step(initial_value.U32(), increment)))
				return Induction {phi, step, initial};
		}
		return {};
	}

	const Inst* BoundedInduction(Value key, const Block* use, const auto& accepts_step,
	                             const auto& accepts_bound,
	                             LaneQuantifier lanes = LaneQuantifier::Any) const {
		const auto induction = MatchInduction(key, accepts_step);
		if (!induction) return {};
		const auto [phi, step, initial_arm] = *induction;
		const auto* increment_block = step->Parent();
		// Structurization can forward the native increment through a synthetic continue block.
		const auto* incoming = phi->PhiBlock(initial_arm ^ 1u);
		const auto* next = phi->Parent();
		for (size_t edges = 0; incoming != increment_block && edges < m_program.blocks.size(); ++edges) {
			if (incoming == nullptr || incoming->terminator.kind != CFG::TerminatorKind::Branch ||
			    incoming->terminator.true_block != next || incoming->ImmPredecessors().size() != 1u)
				break;
			next = incoming;
			incoming = incoming->ImmPredecessors()[0];
		}
		if (incoming != increment_block) return {};

		const auto stops = [&](const Block* block) { return block == phi->Parent(); };
		for (const auto& use_of_key: phi->Uses()) {
			const auto* compare = use_of_key.user;
			if (use_of_key.operand != 0u ||
			    !accepts_bound(*compare, initial_arm)) continue;
			if (!GuardedOnEntry(use != nullptr ? use : increment_block, stops, [&](const EdgePredicate& edge) {
				return edge.positive && (lanes == LaneQuantifier::Any || edge.lanes == lanes) &&
				       Implies(edge.condition, Value(use_of_key.user));
			})) continue;
			// The image bound and the increment guard are separate obligations: a
			// skipped image alone does not prevent signed induction wraparound.
			const auto [found, inserted] = m_loop_proofs.try_emplace(compare);
			if (inserted) {
				LoopBoundProof proof(m_program, *phi, *compare);
				found->second =
				    GuardedOnEntry(increment_block, stops, [&](const EdgePredicate& edge) {
					    return proof.Excludes(edge.condition, edge.positive);
				    });
			}
			if (found->second) return compare;
		}
		return {};
	}

	uint32_t PostTestLoopMaximum(Value key) const {
		const auto induction = MatchInduction(key, ZeroBased);
		if (!induction) return UINT32_MAX;
		const auto [phi, step, initial_arm] = *induction;
		const auto* incoming = phi->PhiBlock(initial_arm ^ 1u);
		if (incoming == nullptr) return UINT32_MAX;
		for (const auto& use: step->Uses()) {
			const auto* compare = use.user;
			uint32_t bound = 0;
			if (use.operand != 0u ||
			    (compare->GetOpcode() != ValueOpcode::SLessThan32 &&
			     compare->GetOpcode() != ValueOpcode::ULessThan32) ||
			    !ImmediateU32(compare->Arg(1), bound) || bound == 0u ||
			    (compare->GetOpcode() == ValueOpcode::SLessThan32 && bound > INT32_MAX) ||
			    !AvailableBefore(*step, *compare)) continue;
			const auto accepts = [&](const EdgePredicate& edge) {
				return edge.positive && edge.lanes == LaneQuantifier::All &&
				       Implies(edge.condition, Value(const_cast<Inst*>(compare)));
			};
			const auto edge = ConditionalEdge(incoming, phi->Parent());
			if ((edge && accepts(*edge)) || GuardedOnEntry(incoming, [&](const Block* block) {
				// A guard before the increment may belong to the previous iteration.
				return block == phi->Parent() || block == step->Parent();
			}, accepts)) {
				// Starting at zero, next < bound keeps the carried value in range;
				// incrementing bound - 1 cannot overflow either accepted comparison domain.
				return bound - 1u;
			}
		}
		return UINT32_MAX;
	}

	const Inst* BoundedLoop(Value key, const Block* use, const auto& accepts_bound,
	                       LaneQuantifier lanes = LaneQuantifier::Any) const {
		return BoundedInduction(key, use, ZeroBased,
		    [&](const Inst& compare, uint32_t initial) {
			    return (compare.GetOpcode() == ValueOpcode::SLessThan32 ||
			            compare.GetOpcode() == ValueOpcode::ULessThan32) &&
			           accepts_bound(compare, initial);
		    }, lanes);
	}

	const Inst* DescendingLoop(Value key, const Block* use, uint32_t& maximum) const {
		return BoundedInduction(key, use,
		    [&](uint32_t initial, uint32_t step) {
			    maximum = initial;
			    return initial <= INT32_MAX && step == UINT32_MAX;
		    }, [](const Inst& compare, uint32_t) {
			    return compare.GetOpcode() == ValueOpcode::SGreaterThanEqual32 &&
			           compare.Arg(1).Resolve() == Value(0u);
		    });
	}

	const Inst* BoundedLoop(Value key, const Block* use) const {
		return BoundedLoop(key, use, [&](const Inst& compare, uint32_t) {
			return ValidateRuntimeValue(m_program, compare.Arg(1), RuntimeValueType::Integer);
		});
	}

	// A zero-based key that only per-lane data bounds takes the count all uniform zero-based
	// loops in the program share, assuming each lane's count stays within it.
	bool ProgramLoopCount(Value key, DescriptorSource::IndirectDescriptor& indirect) const {
		if (!MatchInduction(key, ZeroBased) || BoundedLoop(key, nullptr) != nullptr) return false;
		const Inst* count = nullptr;
		for (auto* block: m_program.blocks) {
			for (auto& phi: *block) {
				if (phi.GetOpcode() != ValueOpcode::Phi) break;
				const auto* compare = BoundedLoop(Value(&phi), nullptr);
				if (compare == nullptr) continue;
				if (count != nullptr && !EquivalentValue(m_program, compare->Arg(1), count->Arg(1)))
					return false;
				count = compare;
			}
		}
		if (count == nullptr) return false;
		std::printf("Warning: shader 0x%016llx bounds a per-lane table loop by its uniform loops' "
		            "count\n", static_cast<unsigned long long>(m_program.shader_hash));
		indirect.key_count    = count->Arg(1);
		indirect.signed_count = count->GetOpcode() == ValueOpcode::SLessThan32;
		return true;
	}

	Value ActiveValue(Value value, Value guard) const {
		value = value.Resolve();
		while (const auto* select = value.TryInstruction()) {
			if (select->GetOpcode() != ValueOpcode::SelectU32 ||
			    !Implies(guard, select->Arg(0))) break;
			value = select->Arg(1).Resolve();
		}
		return value;
	}

	bool Reaches(const Block* from, const Block* to, const Block* stop = nullptr) const {
		std::vector<const Block*> pending {to};
		for (size_t index = 0; index < pending.size(); ++index) {
			if (pending[index] == stop) continue;
			if (pending[index] == from) return true;
			for (const auto* previous: pending[index]->ImmPredecessors())
				if (std::ranges::find(pending, previous) == pending.end()) pending.push_back(previous);
		}
		return false;
	}

	bool InCycle(const Block* block) const {
		return std::ranges::any_of(block->ImmPredecessors(),
		    [&](const Block* previous) { return Reaches(block, previous); });
	}

	static bool BeforeInBlock(const Inst& before, const Inst& after) {
		if (before.Parent() != after.Parent()) return false;
		for (const auto& inst: *before.Parent()) {
			if (&inst == &before) return true;
			if (&inst == &after) return false;
		}
		return false;
	}

	bool Dominates(const Block* initial, const Block* use, Value witness = {}) const {
		// The proven local-zero invocation fails every all-lane bypass condition that excludes it,
		// such as an all-inactive or an all-nonzero bypass.
		const auto* negated = SimplifyGuard(witness).TryInstruction();
		const bool nonzero = negated != nullptr && negated->GetOpcode() == ValueOpcode::LogicalNot;
		std::vector<const Block*> pending {use};
		for (size_t index = 0; index < pending.size(); ++index) {
			const auto* current = pending[index];
			if (current == initial) continue;
			if (current->ImmPredecessors().empty()) return false;
			for (const auto* previous: current->ImmPredecessors()) {
				if (!witness.IsEmpty()) {
					const auto edge = ConditionalEdge(previous, current);
					if (edge && edge->lanes == LaneQuantifier::All &&
					    (edge->positive ? nonzero && Implies(edge->condition, negated->Arg(0))
					                    : Implies(witness, edge->condition))) continue;
				}
				if (std::ranges::find(pending, previous) == pending.end()) pending.push_back(previous);
			}
		}
		return true;
	}

	bool AvailableBefore(const Inst& initial, const Inst& use, Value witness = {}) const {
		return initial.Parent() == use.Parent() ? BeforeInBlock(initial, use)
		                                      : Dominates(initial.Parent(), use.Parent(), witness);
	}

	std::optional<std::array<uint64_t, 5>> LocalIndex(Value value, Value active,
	                                                Value& induction, bool allow_induction) const {
		return AffineIndex<5>(value, [&](Value at) { return ActiveValue(at, active); },
		    [&](Value at) -> std::optional<size_t> {
			    const auto* inst = at.TryInstruction();
			    if (inst == nullptr) return {};
			    uint32_t bits = 32u;
			    if (inst->GetOpcode() == ValueOpcode::BitFieldUExtract &&
			        inst->Arg(1).Resolve() == Value(0u) &&
			        ImmediateU32(inst->Arg(2), bits) && bits <= 32u) {
				    inst = inst->Arg(0).Resolve().TryInstruction();
				    if (inst == nullptr) return {};
			    }
			    uint32_t axis = 0;
			    if (inst->GetOpcode() == ValueOpcode::GetBuiltin &&
			        inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)) &&
			        ImmediateU32(inst->Arg(1), axis) && axis < 3u &&
			        m_input.compute->threads_num[axis] <= (uint64_t {1} << bits)) return axis + 1u;
			    if (!allow_induction || inst->GetOpcode() != ValueOpcode::Phi ||
			        (!induction.IsEmpty() && induction != at)) return {};
			    induction = at;
			    return 4u;
		    });
	}

	bool LinearLocalIndex(const std::array<uint64_t, 5>& index) const {
		uint64_t stride = 1u;
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			if (m_input.compute->threads_num[axis] != 1u && index[axis + 1u] != stride) return false;
			stride *= m_input.compute->threads_num[axis];
		}
		return index[0] == 0u;
	}

	void ForwardPrivateLds() {
		if (m_program.stage != ShaderType::Compute) return;
		uint64_t bytes = 4u;
		for (const auto size: m_input.compute->threads_num) {
			if (size == 0u || size > 1024u) return;
			bytes *= size;
		}
		if (bytes > uint64_t {m_input.compute->lds_size_dwords} * 4u) return;
		struct Access { Inst* inst; uint32_t offset; };
		std::vector<Access> reads, writes;
		Value address;
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				const auto shared = SharedAccessOf(op);
				if (shared == SharedAccess::None &&
				    AddressOpcodeInfoOf(op).access != AddressAccess::Write) continue;
				const auto& memory = m_program.memory_info.at(inst.Flags<MemoryFlags>().index);
				if (memory.kind == ResourceKind::Flat || memory.kind == ResourceKind::FlatLocal) return;
				if (memory.kind != ResourceKind::Lds) continue;
				if ((op != ValueOpcode::LoadSharedU32 && op != ValueOpcode::WriteSharedU32) ||
				    memory.data_bits != 32u || memory.secondary_offset != 0u ||
				    (memory.offset & 3u) != 0u ||
				    uint64_t {memory.offset} + bytes > uint64_t {m_input.compute->lds_size_dwords} * 4u)
					return;
				if (address.IsEmpty()) address = inst.Arg(0).Resolve();
				else if (!EquivalentValue(m_program, address, inst.Arg(0))) return;
				(op == ValueOpcode::LoadSharedU32 ? reads : writes).push_back({&inst, memory.offset});
			}
		}
		Value induction;
		auto index = LocalIndex(address, Value(true), induction, false);
		if (!index) return;
		for (auto& coefficient: *index) {
			if ((coefficient & 3u) != 0u) return;
			coefficient /= 4u;
		}
		if (!LinearLocalIndex(*index)) return;
		// Each invocation owns one DWORD in each disjoint slot. A unique dominating
		// writer therefore survives loops and barriers without cross-lane interference.
		for (const auto& store: writes) {
			if (InCycle(store.inst->Parent()) || std::ranges::any_of(writes, [&](const Access& other) {
				return other.inst != store.inst && uint64_t {other.offset} < store.offset + bytes &&
				       uint64_t {store.offset} < other.offset + bytes;
			})) continue;
			for (const auto& load: reads) {
				if (load.offset != store.offset || !AvailableBefore(*store.inst, *load.inst) ||
				    !Implies(load.inst->Arg(1), store.inst->Arg(2))) continue;
				auto* block = load.inst->Parent();
				const auto where = std::ranges::find_if(block->Instructions(),
				    [&](const Inst& at) { return &at == load.inst; });
				const auto guard = load.inst->Arg(1);
				const auto value = ActiveValue(store.inst->Arg(1), guard);
				auto& forwarded = *block->PrependNewInst(where, ValueOpcode::SelectU32,
				    {guard, value, Value(0u)});
				load.inst->ReplaceUsesWith(Value(&forwarded));
			}
		}
	}

	bool LocalZeroWitness(Value guard) const {
		const auto* equal = SimplifyGuard(guard).TryInstruction();
		const bool negated = equal != nullptr && equal->GetOpcode() == ValueOpcode::LogicalNot;
		if (negated) equal = SimplifyGuard(equal->Arg(0)).TryInstruction();
		const auto expected = negated ? ValueOpcode::INotEqual32 : ValueOpcode::IEqual32;
		if (equal == nullptr || equal->GetOpcode() != expected) return false;
		for (uint32_t arg = 0; arg < 2u; ++arg) {
			if (equal->Arg(arg).Resolve() != Value(0u)) continue;
			Value induction;
			const auto index = LocalIndex(equal->Arg(arg ^ 1u), guard, induction, false);
			if (index && LinearLocalIndex(*index)) return true;
		}
		return false;
	}

	Value UnsignedBound(Value guard, Value value, bool positive = true,
	                    bool finite_bitscan = false, Value witness = {}) const {
		const auto* scan = value.Resolve().TryInstruction();
		std::vector<std::pair<Value, bool>> pending {{guard, positive}};
		std::set<std::pair<const Inst*, bool>> visited;
		while (!pending.empty()) {
			const auto [cur_guard, pos] = pending.back();
			pending.pop_back();
			const auto* test = SimplifyGuard(cur_guard).TryInstruction();
			if (test == nullptr || !visited.emplace(test, pos).second) continue;
			const auto op = test->GetOpcode();
			if (op == ValueOpcode::LogicalNot) {
				pending.emplace_back(test->Arg(0), !pos);
				continue;
			}
			if ((op == ValueOpcode::LogicalAnd && pos) ||
			    (op == ValueOpcode::LogicalOr && !pos)) {
				pending.emplace_back(test->Arg(1), pos);
				pending.emplace_back(test->Arg(0), pos);
				continue;
			}
			if (op == ValueOpcode::LogicalAnd && !pos && !witness.IsEmpty()) {
				// !(A & Q) proves !Q only for accesses whose EXEC also proves A.
				for (uint32_t arm = 2u; arm-- > 0;) {
					if (Implies(witness, test->Arg(arm))) {
						pending.emplace_back(test->Arg(arm ^ 1u), false);
					}
				}
				continue;
			}
			if ((pos ? op == ValueOpcode::ULessThan32 : op == ValueOpcode::UGreaterThanEqual32) &&
			    EquivalentValue(m_program, test->Arg(0), value)) {
				return test->Arg(1);
			}
			if ((pos ? op == ValueOpcode::UGreaterThan32 : op == ValueOpcode::ULessThanEqual32) &&
			    EquivalentValue(m_program, test->Arg(1), value)) {
				return test->Arg(0);
			}
			uint32_t limit = 0;
			if ((pos ? op == ValueOpcode::ULessThanEqual32 : op == ValueOpcode::UGreaterThan32) &&
			    ImmediateU32(test->Arg(1), limit) && limit != UINT32_MAX &&
			    EquivalentValue(m_program, test->Arg(0), value)) {
				return Value(limit + 1u);
			}
			if (finite_bitscan && scan != nullptr && scan->GetOpcode() == ValueOpcode::FindILsb32 &&
			    (pos ? op == ValueOpcode::INotEqual32 : op == ValueOpcode::IEqual32)) {
				for (uint32_t arg = 0; arg < 2u; ++arg) {
					uint32_t excluded;
					if (!ImmediateU32(test->Arg(arg), excluded)) continue;
					if ((excluded == UINT32_MAX &&
					     EquivalentValue(m_program, test->Arg(arg ^ 1u), value)) ||
					    (excluded == 0u &&
					     EquivalentValue(m_program, test->Arg(arg ^ 1u), scan->Arg(0)))) {
						return Value(static_cast<uint32_t>(
						    std::bit_width(SelectorMaximum(scan->Arg(0)))));
					}
				}
			}
		}
		return {};
	}

	Value UnsignedBoundOnEntry(Value key, const Block* block, bool finite_bitscan,
	                           Value witness, const auto& accepts_bound) const {
		const auto* definition = key.Resolve().TryInstruction();
		// A mask-nonzero guard may precede FF1. Its proof lasts until the mask's
		// definition, including a loop Phi that changes it for the next iteration.
		if (definition != nullptr && definition->GetOpcode() == ValueOpcode::FindILsb32)
			definition = definition->Arg(0).Resolve().TryInstruction();
		Value bound;
		const bool guarded = GuardedOnEntry(block, [&](const Block* current) {
			return definition != nullptr && current == definition->Parent();
		}, [&](const EdgePredicate& edge) {
			if (edge.lanes != LaneQuantifier::All) return false;
			const auto candidate = UnsignedBound(edge.condition, key, edge.positive, finite_bitscan, witness);
			// Every incoming path must prove the same bound for this key definition.
			if (candidate.IsEmpty() || (!bound.IsEmpty() &&
			    !EquivalentValue(m_program, candidate, bound)) || !accepts_bound(candidate)) return false;
			bound = candidate;
			return true;
		});
		return guarded ? bound : Value {};
	}

	Value UnsignedBoundOnEntry(Value key, const Block* block, bool finite_bitscan = false,
	                           Value witness = {}) const {
		return UnsignedBoundOnEntry(key, block, finite_bitscan, witness, [&](Value bound) {
			return ValidateRuntimeValue(m_program, bound, RuntimeValueType::Integer);
		});
	}

	Value UnsignedResourceBound(Value key, const Inst& handle) const {
		Value bound;
		for (const auto& use: handle.Uses()) {
			const auto& access = *use.user;
			const auto op = access.GetOpcode();
			if (use.operand != 0u || (op != ValueOpcode::ImageSampleRaw &&
			    op != ValueOpcode::ImageGatherRaw && op != ValueOpcode::ImageRead &&
			    !IndirectBufferAccess(access))) return {};
			// The access's own EXEC predicate proves the range for every accessing lane;
			// an EXEC-nonzero branch alone only proves that some lane is active.
			const auto active = access.Arg(access.NumArgs() - 1u);
			auto candidate = UnsignedBound(active, key);
			if (candidate.IsEmpty()) candidate = UnsignedBoundOnEntry(key, access.Parent(), false, active);
			if (candidate.IsEmpty()) {
				// A waterfall key inherits the local value's bound only where this same
				// EXEC predicate proves both their equality and the local range. Otherwise
				// the key itself is bounded, with this predicate witnessing its own !(A & Q).
				const auto local = EqualLocalKey(active, key);
				candidate = UnsignedBound(active, local.IsEmpty() ? key : local, true, false, active);
			}
			if (candidate.IsEmpty() ||
			    !ValidateRuntimeValue(m_program, candidate, RuntimeValueType::Integer)) return {};
			if (bound.IsEmpty()) bound = candidate;
			else if (!EquivalentValue(m_program, candidate, bound)) return {};
		}
		return bound;
	}

	Value SharedLoadBound(Value value, Value active) {
		const auto* read = ActiveValue(value, active).TryInstruction();
		if (read == nullptr || read->GetOpcode() != ValueOpcode::LoadSharedU32 ||
		    m_program.stage != ShaderType::Compute ||
		    !Implies(active, read->Arg(1))) return {};
		uint64_t group_size = 1u;
		for (const auto dimension: m_input.compute->threads_num) {
			if (dimension == 0u || dimension > m_input.compute->wave_size) return {};
			group_size *= dimension;
		}
		// Indexed LDS operations within one native wave execute in order. Multiwave
		// append lists require a separate workgroup synchronization proof.
		if (group_size != m_input.compute->wave_size || group_size != m_program.wave_size ||
		    (group_size != 32u && group_size != 64u)) return {};
		const auto memory = [&](const Inst& inst) -> const MemoryInfo* {
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= m_program.memory_info.size()) return nullptr;
			const auto& info = m_program.memory_info[index];
			return info.kind == ResourceKind::Lds && info.data_bits == 32u &&
			       info.secondary_offset == 0u && info.component_index == 0u ? &info : nullptr;
		};
		const auto constant_address = [&](const Inst& inst, uint32_t& address) {
			const auto* info = memory(inst);
			if (info == nullptr || !ImmediateU32(ActiveValue(inst.Arg(0), inst.Arg(inst.NumArgs() - 1u)), address) ||
			    uint64_t {address} + info->offset > UINT32_MAX) return false;
			address += info->offset;
			return (address & 3u) == 0u;
		};
		const auto* read_memory = memory(*read);
		AffineOffset read_offset;
		if (read_memory == nullptr || !MatchAffineOffset(read->Arg(0), active, read_offset) ||
		    read_offset.stride != 4u || read_offset.offset + read_memory->offset > UINT32_MAX) return {};
		const auto* limit = BoundedLoop(read_offset.index, read->Parent(),
		    [](const Inst& compare, uint32_t) { return compare.GetOpcode() == ValueOpcode::ULessThan32; });
		if (limit == nullptr) return {};
		auto counter_active = active;
		const auto* counter_read = ActiveValue(limit->Arg(1), active).TryInstruction();
		// Every lane reads the same counter word, so a selected lane carries it.
		if (counter_read != nullptr && counter_read->GetOpcode() == ValueOpcode::ReadFirstLane) {
			counter_active = counter_read->Arg(1);
			counter_read = ActiveValue(counter_read->Arg(0), counter_active).TryInstruction();
		}
		uint32_t consumed_counter = 0;
		if (counter_read == nullptr || counter_read->GetOpcode() != ValueOpcode::LoadSharedU32 ||
		    !Implies(counter_active, counter_read->Arg(1)) ||
		    !constant_address(*counter_read, consumed_counter)) return {};

		struct Initializer { const Inst* inst; uint32_t address; uint32_t words; };
		struct Append { const Inst* atomic; const Inst* store = nullptr; uint32_t counter; uint32_t base = 0; Value bound; };
		std::vector<Initializer> initializers;
		std::vector<Append> appends;
		std::vector<const Inst*> stores;
		for (const auto* block: m_program.blocks) {
			for (const auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if (AddressOpcodeInfoOf(op).access == AddressAccess::Write) {
					const auto index = inst.Flags<MemoryFlags>().index;
					if (index >= m_program.memory_info.size() ||
					    m_program.memory_info[index].kind == ResourceKind::Flat ||
					    m_program.memory_info[index].kind == ResourceKind::FlatLocal) return {};
				}
				const auto access = SharedAccessOf(op);
				if (access == SharedAccess::None || access == SharedAccess::Read) continue;
				const auto* info = memory(inst);
				if (info == nullptr) return {};
				const auto guard = inst.Arg(inst.NumArgs() - 1u);
				uint32_t address = 0;
				if (access == SharedAccess::Atomic) {
					if (op != ValueOpcode::SharedAtomicIAdd32 ||
					    ActiveValue(inst.Arg(1), guard) != Value(1u) ||
					    !constant_address(inst, address) ||
					    uint64_t {address} + 4u > uint64_t {m_input.compute->lds_size_dwords} * 4u ||
					    address >= 0x10000u ||
					    std::ranges::any_of(appends, [&](const Append& at) { return at.counter == address; })) return {};
					appends.push_back({.atomic = &inst, .counter = address});
				} else if (access == SharedAccess::Write) {
					const auto words = SharedComponentCount(op);
					bool zero = LocalZeroWitness(guard) && constant_address(inst, address) &&
					            address % (words * 4u) == 0u;
					for (uint32_t word = 0; word < words; ++word)
						zero = zero && ActiveValue(inst.Arg(word + 1u), guard) == Value(0u);
					if (zero) initializers.push_back({&inst, address, words});
					else if (op == ValueOpcode::WriteSharedU32) stores.push_back(&inst);
					else return {};
				} else return {};
			}
		}
		if (appends.empty() || stores.size() != appends.size() || initializers.empty()) return {};
		for (const auto& initial: initializers) {
			if (InCycle(initial.inst->Parent())) return {};
			// Every initialized word is a counter; arbitrary shared writes are not append provenance.
			for (uint32_t word = 0; word < initial.words; ++word)
				if (std::ranges::none_of(appends, [&](const Append& at) {
					return uint64_t {initial.address} + word * 4u == at.counter;
				})) return {};
		}
		for (auto& append: appends) {
			const auto guard = append.atomic->Arg(2);
			if (Reaches(counter_read->Parent(), append.atomic->Parent())) return {};
			uint32_t initialized = 0;
			for (const auto& initial: initializers) {
				const auto witness = initial.inst->Arg(initial.inst->NumArgs() - 1u);
				if (!AvailableBefore(*initial.inst, *append.atomic, witness) ||
				    !AvailableBefore(*initial.inst, *counter_read, witness)) return {};
				if (append.counter >= initial.address &&
				    uint64_t {append.counter} < uint64_t {initial.address} + initial.words * 4u) ++initialized;
			}
			if (initialized != 1u) return {};
			for (const auto* store: stores) {
				if (!EquivalentValue(m_program, guard, store->Arg(2)) ||
				    !BeforeInBlock(*append.atomic, *store)) continue;
				AffineOffset address;
				const auto* info = memory(*store);
				if (!MatchAffineOffset(store->Arg(0), guard, address) ||
				    address.index != Value(const_cast<Inst*>(append.atomic)) || address.stride != 4u ||
				    (address.offset + info->offset) % 4u != 0u ||
				    address.offset + info->offset + uint64_t {MaxIndirectDescriptorProbes} * 4u > UINT32_MAX ||
				    std::ranges::any_of(appends, [&](const Append& at) {
					    return address.offset + info->offset <= at.counter;
				    })) continue;
				if (append.store != nullptr) return {};
				const auto payload = ActiveValue(store->Arg(1), guard);
				Value induction;
				const auto index = LocalIndex(payload, guard, induction, true);
				if (!index || !LinearLocalIndex(*index) || (*index)[4] != 1u || induction.IsEmpty()) return {};
				const auto* bound = BoundedInduction(induction, append.atomic->Parent(),
				    [&](uint32_t initial, uint32_t step) { return initial == 0u && step == group_size; },
				    [&](const Inst& compare, uint32_t initial) {
					    return compare.GetOpcode() == ValueOpcode::ULessThan32 &&
					           ValidateRuntimeValue(m_program, compare.Arg(1), RuntimeValueType::Integer) &&
					           !InCycle(induction.Instruction()->PhiBlock(initial));
				    });
				if (bound == nullptr || !EquivalentValue(m_program, UnsignedBound(guard, payload), bound->Arg(1))) return {};
				// Each re-entry must advance the producer induction; an inner append
				// cycle could otherwise increment the counter twice for one candidate.
				if (std::ranges::any_of(append.atomic->Parent()->ImmPredecessors(), [&](const Block* previous) {
					    return Reaches(append.atomic->Parent(), previous, induction.Instruction()->Parent());
				    })) return {};
				append.store = store;
				append.base = static_cast<uint32_t>(address.offset + info->offset);
				append.bound = bound->Arg(1);
			}
			if (append.store == nullptr) return {};
		}
		const auto consumed = std::ranges::find_if(appends, [&](const Append& at) {
			return at.counter == consumed_counter && at.base == read_offset.offset + read_memory->offset;
		});
		if (consumed == appends.end()) return {};
		const auto emit = [&](ValueOpcode op, std::initializer_list<Value> args) {
			auto& node = m_program.value_storage.emplace_back(op);
			size_t index = 0;
			for (const auto arg: args) node.SetArg(index++, arg);
			return Value(&node);
		};
		Value bound = consumed->bound;
		Value all_bounds = bound;
		for (const auto& append: appends) {
			if (EquivalentValue(m_program, consumed->bound, append.bound)) continue;
			all_bounds = emit(ValueOpcode::UMax32, {all_bounds, append.bound});
			auto contribution = append.bound;
			if (append.base != consumed->base) {
				// Each list occupies at most its producer bound in DWORDs. Only a list
				// whose possible payload span overlaps the consumed span can supply its values.
				const auto gap = (std::max(append.base, consumed->base) -
				                  std::min(append.base, consumed->base)) / 4u;
				const auto overlap = emit(ValueOpcode::ULessThan32,
				    {Value(gap), append.base < consumed->base ? append.bound : consumed->bound});
				contribution = emit(ValueOpcode::SelectU32, {overlap, contribution, Value(0u)});
			}
			bound = emit(ValueOpcode::UMax32, {bound, contribution});
		}
		if (all_bounds == consumed->bound) return bound;
		bound = emit(ValueOpcode::SelectU32,
		    {emit(ValueOpcode::ULessThan32, {Value(0u), consumed->bound}), bound, Value(0u)});
		// Validate even disjoint producers: the address and counter proofs require
		// every bound to fit the existing cap, including when the consumer is empty.
		return emit(ValueOpcode::SelectU32,
		    {emit(ValueOpcode::ULessThanEqual32, {all_bounds, Value(MaxIndirectDescriptorProbes)}),
		     bound, all_bounds});
	}

	bool MatchCountedRows(const Inst& read, DescriptorSource& count_source, uint32_t& row_stride) {
		Value element;
		uint32_t offset = 0, stride = 0;
		if (!MatchTableOffset(read.Arg(1), element, offset, stride) || offset != 0u || stride != 4u)
			return false;
		const auto* lane = element.TryInstruction();
		const bool selected_lane = lane != nullptr && lane->GetOpcode() == ValueOpcode::ReadFirstLane;
		if (selected_lane) {
			if (!HasActiveLane(lane->Arg(1), lane->Parent())) return false;
			element = ActiveValue(lane->Arg(0), lane->Arg(1));
		}
		const auto* sum = element.TryInstruction();
		if (sum == nullptr || sum->GetOpcode() != ValueOpcode::IAdd32) return false;
		for (uint32_t arm = 0; arm < 2u; ++arm) {
			const auto index = sum->Arg(arm).Resolve();
			// Enumerating every unsigned index below the count needs the access guard,
			// independently of the loop's initial value, step or termination.
			if (!UnsignedBoundOnEntry(index, read.Parent(), false, {}, [&](Value bound) {
				const auto* count = bound.Resolve().TryInstruction();
				if (count == nullptr || count->GetOpcode() != ValueOpcode::ReadConstBuffer) return false;
				uint32_t memory_index = 0;
				const auto* memory = ScalarReadMemory(*count, memory_index);
				Value row;
				uint32_t count_offset = 0, count_stride = 0;
				if (memory == nullptr || memory->offset != 0u ||
				    !MatchTableOffset(count->Arg(1), row, count_offset, count_stride) ||
				    count_offset != 0u || count_stride != 4u) return false;
				if (selected_lane) {
					const auto* row_lane = row.TryInstruction();
					if (row_lane == nullptr || row_lane->GetOpcode() != ValueOpcode::ReadFirstLane ||
					    !ValidateWaveUniformValue(m_program,
					        EquivalentValue(m_program, row_lane->Arg(1), lane->Arg(1))
					            ? row : row_lane->Arg(0))) return false;
					// Equal EXEC masks select the same row; differing masks need a globally
					// uniform row to keep count and key addresses correlated.
					row = ActiveValue(row_lane->Arg(0), lane->Arg(1));
				}
				const auto affine = AffineIndex<3>(element,
				    [&](Value value) { return selected_lane ? ActiveValue(value, lane->Arg(1)) : value.Resolve(); },
				    [&](Value value) -> std::optional<size_t> {
					    if (value == index) return 1u;
					    if (EquivalentValue(m_program, value, row)) return 2u;
					    return {};
				    });
				if (!affine || (*affine)[0] != 0u || (*affine)[1] != 1u ||
				    (*affine)[2] == 0u || (*affine)[2] > UINT32_MAX / 4u ||
				    !MakeRuntimeTableSource(*count, count_source)) return false;
				// Count addresses identify rows modulo 2^30. A DWORD-aligned row
				// spacing preserves those aliases in the U32 selector address too.
				row_stride = static_cast<uint32_t>((*affine)[2] * 4u);
				return true;
			}).IsEmpty()) return true;
		}
		return false;
	}

	bool MatchScalarMaterialKey(Value key, DescriptorSource::IndirectDescriptor& indirect,
	                            DescriptorSource& material_source, bool require_bound,
	                            bool allow_scalar_domain = false) {
		DescriptorSource::IndirectDescriptor::SelectorRead selector;
		const auto* read = SelectorReadValue(key, Value(true), selector);
		if (read == nullptr) return false;
		uint32_t memory_index = 0;
		const auto* memory = ScalarReadMemory(*read, memory_index);
		Value index;
		Value count;
		Value first(0u);
		uint32_t workgroup_axis = UINT32_MAX;
		DescriptorSource count_source;
		uint32_t row_stride = 0;
		uint32_t offset = 0, stride = 0;
		if (memory == nullptr || memory->offset > INT32_MAX ||
		    !MemoryIndexBelongsTo(memory_index, *read)) return false;
		const bool affine = MatchTableOffset(read->Arg(1), index, offset, stride) &&
		    ((offset | stride | memory->offset) & 3u) == 0u &&
		    uint64_t {offset} + memory->offset <= UINT32_MAX;
		if (affine) {
			workgroup_axis = WorkgroupAxis(index, &first);
			const auto* bound = BoundedLoop(index, read->Parent());
			if (bound != nullptr && bound->GetOpcode() == ValueOpcode::ULessThan32) {
				count = bound->Arg(1);
			} else if (workgroup_axis == UINT32_MAX && !require_bound) {
				// Other bounded indices, such as bit scans, read the records up to their maximum;
				// a bit scan of a nonzero mask load reads only the ranks its words set.
				const auto maximum = SelectorMaximum(index);
				if (maximum == 31u) selector.ranks = BitScanRanks(index);
				if (maximum < MaxIndirectDescriptorProbes) count = Value(maximum + 1u);
			}
			// A record selected from a shared append list is below the list's bound.
			const auto* lane = index.Resolve().TryInstruction();
			if (count.IsEmpty() && lane != nullptr &&
			    lane->GetOpcode() == ValueOpcode::ReadFirstLane &&
			    HasActiveLane(lane->Arg(1), lane->Parent()))
				count = SharedLoadBound(lane->Arg(0), lane->Arg(1));
		}
		// Raw pointers have no V# bounds for enumerating an unbounded selector domain.
		if (memory->kind == ResourceKind::ScalarAddress && count.IsEmpty() &&
		    workgroup_axis == UINT32_MAX) return false;
		const bool rows = count.IsEmpty() && workgroup_axis == UINT32_MAX && memory->offset == 0u &&
		    MatchCountedRows(*read, count_source, row_stride);
		if (rows) {
			stride = 4u;
			offset = 0u;
		} else if (count.IsEmpty() && workgroup_axis == UINT32_MAX &&
		           !require_bound && allow_scalar_domain && affine && stride != 0u) {
			// The bounded selector V# covers this complete U32 address congruence
			// class, including the zero value of out-of-bounds scalar reads.
			stride = 1u << std::countr_zero(stride);
			offset %= stride;
		} else if (count.IsEmpty() && workgroup_axis == UINT32_MAX) {
			auto address = read->Arg(1);
			auto active = Value(true);
			const auto* lane = address.Resolve().TryInstruction();
			const bool selected_lane = lane != nullptr && lane->GetOpcode() == ValueOpcode::ReadFirstLane;
			if (selected_lane) {
				active = lane->Arg(1);
				address = lane->Arg(0);
			}
			AffineOffset affine;
			if (!MatchAffineOffset(address, active, affine) ||
			    affine.index.IsEmpty() || affine.stride == 0u) return false;
			if (!selected_lane) lane = affine.index.Resolve().TryInstruction();
			if (lane == nullptr || lane->GetOpcode() != ValueOpcode::ReadFirstLane ||
			    !HasActiveLane(lane->Arg(1), lane->Parent())) return false;
			if (selected_lane) count = SharedLoadBound(affine.index, active);
			if (!count.IsEmpty() && ((affine.stride | affine.offset) & 3u) == 0u &&
			    affine.offset + memory->offset <= UINT32_MAX) {
				stride = static_cast<uint32_t>(affine.stride);
				offset = static_cast<uint32_t>(affine.offset);
			} else {
				count = {};
				// Without a bounded source, U32 multiplication's complete offset domain
				// is one congruence class rather than the V#'s structured record stride.
				const auto divisor = std::gcd(affine.stride, uint64_t {1} << 32u);
				stride = static_cast<uint32_t>(std::max<uint64_t>(divisor, 4u));
				offset = static_cast<uint32_t>(affine.offset % divisor) & ~3u;
			}
		}
		// A bounded image heap already supports GPU selection. Only enumerate a
		// narrower proven selector domain instead of scanning unrelated material data.
		if (require_bound && count.IsEmpty() && workgroup_axis == UINT32_MAX && !rows) return false;
		if (!MakeRuntimeTableSource(*read, material_source)) return false;
		selector.source = InternSource(material_source);
		selector.stride = stride;
		selector.offset = offset + (memory->offset & ~3u);
		indirect.selector = std::move(selector);
		indirect.key_count = count;
		indirect.workgroup_axis = workgroup_axis;
		if (rows) indirect.selector->rows.emplace(
		    DescriptorSource::IndirectDescriptor::SelectorRead::CountedRows {
		        .source = InternSource(count_source), .stride = row_stride});
		if (!count.IsEmpty() || workgroup_axis != UINT32_MAX) indirect.selector_first = first;
		return true;
	}

	void FoldBoundedLoopSelectors() {
		const auto fold = [&](const Inst& compare, const auto& bounds, const Block* header) {
			for (auto* block: m_program.blocks) {
				if (!GuardedOnEntry(block, [&](const Block* at) { return at == header; },
				    [&](const EdgePredicate& edge) {
					    return edge.positive && Implies(edge.condition, Value(const_cast<Inst*>(&compare)));
				    })) continue;
				for (auto& inst: *block)
					if (inst.GetOpcode() == ValueOpcode::SelectU32 &&
					    SelectorPredicateFalse(inst.Arg(0), bounds)) inst.ReplaceUsesWith(inst.Arg(2));
			}
		};
		for (auto* header: m_program.blocks) {
			for (auto& induction: *header) {
				const Inst* carried = nullptr;
				uint32_t maximum = 0;
				if (const auto* compare = DescendingLoop(Value(&induction), nullptr, maximum)) {
					const std::array bounds {std::pair {static_cast<const Inst*>(&induction), maximum}};
					fold(*compare, bounds, header);
					continue;
				}
				const auto* compare = BoundedLoop(Value(&induction), nullptr,
				    [&](const Inst& test, uint32_t initial) {
					if (test.GetOpcode() != ValueOpcode::ULessThan32) return false;
					carried = test.Arg(1).Resolve().TryInstruction();
					if (carried == nullptr || carried->GetOpcode() != ValueOpcode::Phi ||
					    carried->GetType() != Type::U32 || carried->Parent() != header ||
					    carried->NumArgs() != 2u || carried->NumPhiBlocks() != 2u ||
					    carried->PhiBlock(0) != induction.PhiBlock(0) ||
					    carried->PhiBlock(1) != induction.PhiBlock(1)) return false;
					maximum = SelectorMaximum(carried->Arg(initial));
					if (maximum == 0u || maximum == UINT32_MAX) return false;
					const std::array assumptions {std::pair {static_cast<const Inst*>(&induction), maximum - 1u},
					                             std::pair {carried, maximum}};
					// Close the carried-bound invariant before using either assumption.
					return SelectorMaximum(carried->Arg(initial ^ 1u), assumptions) <= maximum;
				});
				if (compare == nullptr) continue;
				const std::array assumptions {std::pair {static_cast<const Inst*>(&induction), maximum - 1u},
				                             std::pair {carried, maximum}};
				// Native scalar branches advance the zero/+1 induction uniformly, so
				// even an any-lane bound test limits it by the largest carried bound.
				// Per-lane edges instead require the comparison on every executing lane.
				fold(*compare, assumptions, header);
			}
		}
		FoldInductiveSelectors();
	}

	void FoldInductiveSelectors() {
		struct Range {
			uint32_t low  = 0u;
			uint32_t high = UINT32_MAX;

			bool operator==(const Range&) const = default;
		};
		struct GuardState {
			std::vector<const Inst*> pos;
			std::vector<const Inst*> neg;
			bool                     contradictory = false;

			auto operator<=>(const GuardState&) const = default;
		};

		std::unordered_map<const Block*, size_t> block_index;
		block_index.reserve(m_program.blocks.size());
		for (size_t i = 0; i < m_program.blocks.size(); ++i) {
			block_index.emplace(m_program.blocks[i], i);
		}
		std::unordered_map<const Inst*, uint32_t> inst_order;
		uint32_t                                  next_order = 0u;
		const auto                                order_of   = [&](const Inst* inst) -> uint32_t {
			const auto [it, inserted] = inst_order.try_emplace(inst, next_order);
			if (inserted) ++next_order;
			return it->second;
		};
		const auto order_less = [&](const Inst* a, const Inst* b) {
			return order_of(a) < order_of(b);
		};
		const auto is_selector_pred = [&](auto&& self, Value cond, uint32_t depth = 0u) -> bool {
			if (depth > 8u) return false;
			const auto* inst = cond.Resolve().TryInstruction();
			if (inst == nullptr) return false;
			order_of(inst);
			const auto op = inst->GetOpcode();
			if (op == ValueOpcode::LogicalAnd) {
				return self(self, inst->Arg(0), depth + 1u) || self(self, inst->Arg(1), depth + 1u);
			}
			uint32_t imm = 0u;
			if (op == ValueOpcode::IEqual32) {
				return ImmediateU32(inst->Arg(0), imm) || ImmediateU32(inst->Arg(1), imm);
			}
			if (op == ValueOpcode::ULessThan32) {
				return ImmediateU32(inst->Arg(0), imm);
			}
			return false;
		};

		std::vector<Inst*> candidate_selects;
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				order_of(&inst);
				if (inst.GetOpcode() == ValueOpcode::SelectU32 &&
				    is_selector_pred(is_selector_pred, inst.Arg(0))) {
					candidate_selects.push_back(&inst);
				}
			}
		}
		if (candidate_selects.empty()) return;

		// Precompute exact Dominates bitset matching Tracker::Dominates(def, use).
		const size_t n_blocks = m_program.blocks.size();
		const size_t words    = (n_blocks + 63u) / 64u;
		std::vector<std::vector<uint64_t>> dom_bits(n_blocks, std::vector<uint64_t>(words, ~uint64_t {0}));
		if (words > 0u) {
			const uint32_t rem       = static_cast<uint32_t>(n_blocks % 64u);
			const uint64_t last_mask = rem == 0u ? ~uint64_t {0} : (uint64_t {1} << rem) - 1u;
			for (size_t i = 0; i < n_blocks; ++i) {
				dom_bits[i].back() = last_mask;
				if (m_program.blocks[i]->ImmPredecessors().empty()) {
					std::ranges::fill(dom_bits[i], uint64_t {0});
					dom_bits[i][i / 64u] = uint64_t {1} << (i % 64u);
				}
			}
			bool dom_changed = true;
			while (dom_changed) {
				dom_changed = false;
				for (size_t i = 0; i < n_blocks; ++i) {
					const auto& preds = m_program.blocks[i]->ImmPredecessors();
					if (preds.empty()) continue;
					std::vector<uint64_t> next(words, ~uint64_t {0});
					next.back() = last_mask;
					for (const auto* pred: preds) {
						const auto pit = block_index.find(pred);
						if (pit == block_index.end()) continue;
						for (size_t w = 0; w < words; ++w) next[w] &= dom_bits[pit->second][w];
					}
					next[i / 64u] |= uint64_t {1} << (i % 64u);
					if (next != dom_bits[i]) {
						dom_bits[i] = std::move(next);
						dom_changed = true;
					}
				}
			}
		}
		const auto dominates_fast = [&](const Block* def_blk, const Block* use_blk) {
			if (def_blk == nullptr || use_blk == nullptr) return false;
			if (def_blk == use_blk) return true;
			const auto dit = block_index.find(def_blk);
			const auto uit = block_index.find(use_blk);
			if (dit == block_index.end() || uit == block_index.end()) return false;
			return (dom_bits[uit->second][dit->second / 64u] &
			        (uint64_t {1} << (dit->second % 64u))) != 0u;
		};
		const auto strictly_dominates = [&](const Block* def_blk, const Block* use_blk) {
			return def_blk != use_blk && dominates_fast(def_blk, use_blk);
		};

		std::map<std::pair<const Inst*, const Inst*>, bool> equiv_cache;
		const auto equiv = [&](auto&& self, Value a, Value b, uint32_t depth = 0u) -> bool {
			a = a.Resolve();
			b = b.Resolve();
			if (a == b) return true;
			if (depth > 4u) return false;
			const auto* ia = a.TryInstruction();
			const auto* ib = b.TryInstruction();
			if (ia == nullptr || ib == nullptr) return false;
			const auto key = ia < ib ? std::pair {ia, ib} : std::pair {ib, ia};
			if (depth == 0u) {
				if (const auto found = equiv_cache.find(key); found != equiv_cache.end())
					return found->second;
			}
			bool ans = ia->GetOpcode() == ib->GetOpcode() && ia->NumArgs() == ib->NumArgs();
			if (ans && ia->GetOpcode() == ValueOpcode::Phi) {
				ans = ia->Parent() == ib->Parent() && ia->NumPhiBlocks() == ib->NumPhiBlocks();
				for (size_t i = 0; ans && i < ia->NumPhiBlocks(); ++i) {
					ans &= ia->PhiBlock(i) == ib->PhiBlock(i);
				}
			}
			for (size_t i = 0; ans && i < ia->NumArgs(); ++i) {
				ans &= self(self, ia->Arg(i), ib->Arg(i), depth + 1u);
			}
			if (depth == 0u) equiv_cache.emplace(key, ans);
			return ans;
		};

		std::map<std::pair<const Inst*, const Inst*>, bool> implies_cache;
		const auto implies_guard = [&](Value root_guard, Value root_req) -> bool {
			struct ImplyFrame {
				Value       guard;
				Value       req;
				const Inst* g_inst       = nullptr;
				const Inst* r_inst       = nullptr;
				uint8_t     kind         = 0u; // 0=Enter, 1=And, 2=Or, 3=Phi
				size_t      arm          = 0u;
				bool        sub_req      = false;
				bool        non_self     = false;
				bool        dep_on_cycle = false;
				bool        in_visiting  = false;
			};
			std::set<std::pair<const Inst*, const Inst*>> visiting_imply;
			std::vector<ImplyFrame>                         stack {{.guard = root_guard, .req = root_req}};
			bool                                            last_ans   = false;
			bool                                            last_cycle = false;
			const auto finish = [&](ImplyFrame& frame, bool ans, bool hit_cycle) {
				const bool cycle = frame.dep_on_cycle || hit_cycle;
				if (frame.in_visiting) {
					visiting_imply.erase({frame.g_inst, frame.r_inst});
					if (!cycle && frame.g_inst != nullptr && frame.r_inst != nullptr) {
						implies_cache.emplace(std::pair {frame.g_inst, frame.r_inst}, ans);
					}
				}
				last_ans   = ans;
				last_cycle = cycle;
				stack.pop_back();
			};
			while (!stack.empty()) {
				auto& frame = stack.back();
				if (frame.kind == 0u) {
					const auto guard = frame.guard.Resolve();
					const auto req   = frame.req.Resolve();
					if (req == Value(true) || equiv(equiv, guard, req)) {
						finish(frame, true, false);
						continue;
					}
					const auto* g_inst = guard.TryInstruction();
					const auto* r_inst = req.TryInstruction();
					if (g_inst == nullptr) {
						finish(frame, false, false);
						continue;
					}
					frame.guard  = guard;
					frame.req    = req;
					frame.g_inst = g_inst;
					frame.r_inst = r_inst;
					if (r_inst != nullptr) {
						if (const auto found = implies_cache.find({g_inst, r_inst});
						    found != implies_cache.end()) {
							finish(frame, found->second, false);
							continue;
						}
						if (!visiting_imply.emplace(g_inst, r_inst).second) {
							finish(frame, false, true);
							continue;
						}
						frame.in_visiting = true;
					}
					const auto op = g_inst->GetOpcode();
					if (op == ValueOpcode::LogicalAnd) {
						frame.kind = 1u;
						stack.push_back({.guard = g_inst->Arg(0), .req = req});
						continue;
					}
					if (op == ValueOpcode::LogicalOr) {
						frame.kind = 2u;
						stack.push_back({.guard = g_inst->Arg(0), .req = req});
						continue;
					}
					if (op == ValueOpcode::Phi && g_inst->GetType() == Type::U1 &&
					    g_inst->NumArgs() != 0u && g_inst->NumPhiBlocks() == g_inst->NumArgs()) {
						const auto* phi_blk = g_inst->Parent();
						bool        in_loop = false;
						for (size_t i = 0; i < g_inst->NumPhiBlocks(); ++i) {
							if (dominates_fast(phi_blk, g_inst->PhiBlock(i))) {
								in_loop = true;
								break;
							}
						}
						if (!in_loop ||
						    (r_inst != nullptr && strictly_dominates(r_inst->Parent(), phi_blk))) {
							frame.kind = 3u;
							frame.arm  = 0u;
							const bool carried = dominates_fast(phi_blk, g_inst->PhiBlock(0));
							stack.push_back({.guard = g_inst->Arg(0),
							                 .req   = carried ? guard : req});
							continue;
						}
					}
					finish(frame, false, false);
					continue;
				}
				frame.dep_on_cycle |= last_cycle;
				if (frame.kind == 1u) {
					if (last_ans) {
						finish(frame, true, false);
						continue;
					}
					if (frame.arm == 0u) {
						frame.arm = 1u;
						stack.push_back({.guard = frame.g_inst->Arg(1), .req = frame.req});
						continue;
					}
					finish(frame, false, false);
					continue;
				}
				if (frame.kind == 2u) {
					if (!last_ans) {
						finish(frame, false, false);
						continue;
					}
					if (frame.arm == 0u) {
						frame.arm = 1u;
						stack.push_back({.guard = frame.g_inst->Arg(1), .req = frame.req});
						continue;
					}
					finish(frame, true, false);
					continue;
				}
				const auto* g_inst  = frame.g_inst;
				const auto* phi_blk = g_inst->Parent();
				const bool  carried = dominates_fast(phi_blk, g_inst->PhiBlock(frame.arm));
				if (frame.sub_req) {
					frame.sub_req = false;
					if (!last_ans) {
						finish(frame, false, false);
						continue;
					}
					frame.non_self = true;
				} else if (carried && !last_ans) {
					frame.sub_req = true;
					stack.push_back({.guard = g_inst->Arg(frame.arm), .req = frame.req});
					continue;
				} else if (!carried) {
					if (!last_ans) {
						finish(frame, false, false);
						continue;
					}
					frame.non_self = true;
				}
				if (++frame.arm < g_inst->NumArgs()) {
					const bool next_carr = dominates_fast(phi_blk, g_inst->PhiBlock(frame.arm));
					stack.push_back({.guard = g_inst->Arg(frame.arm),
					                 .req   = next_carr ? frame.guard : frame.req});
					continue;
				}
				finish(frame, frame.non_self, false);
			}
			return last_ans;
		};

		std::map<std::pair<std::vector<const Inst*>, const Block*>, std::vector<const Inst*>>
		    inv_cache;
		const auto invariant_guards = [&](std::span<const Inst* const> pos,
		                                  const Block* target_blk) -> std::vector<const Inst*> {
			std::vector<const Inst*> key_vec(pos.begin(), pos.end());
			const auto               key = std::pair {key_vec, target_blk};
			if (const auto found = inv_cache.find(key); found != inv_cache.end()) {
				return found->second;
			}
			std::vector<const Inst*>        cand;
			std::unordered_set<const Inst*> seen;
			std::vector<const Inst*>        pending(pos.rbegin(), pos.rend());
			while (!pending.empty()) {
				const auto* g = pending.back();
				pending.pop_back();
				if (g == nullptr || !seen.insert(g).second) continue;
				if (strictly_dominates(g->Parent(), target_blk)) {
					cand.push_back(g);
					continue;
				}
				const auto op = g->GetOpcode();
				if (op == ValueOpcode::LogicalAnd) {
					pending.push_back(g->Arg(1).Resolve().TryInstruction());
					pending.push_back(g->Arg(0).Resolve().TryInstruction());
				} else if (op == ValueOpcode::Phi && g->NumArgs() != 0u &&
				           g->NumPhiBlocks() == g->NumArgs()) {
					for (size_t i = 0; i < g->NumArgs(); ++i) {
						if (dominates_fast(g->Parent(), g->PhiBlock(i))) continue;
						std::vector<const Inst*>        sub;
						std::unordered_set<const Inst*> sub_seen;
						std::vector<const Inst*>        sub_pending {
                            g->Arg(i).Resolve().TryInstruction()};
						while (!sub_pending.empty()) {
							const auto* x = sub_pending.back();
							sub_pending.pop_back();
							if (x == nullptr || !sub_seen.insert(x).second) continue;
							if (strictly_dominates(x->Parent(), target_blk)) {
								sub.push_back(x);
								continue;
							}
							const auto xop = x->GetOpcode();
							if (xop == ValueOpcode::LogicalAnd) {
								sub_pending.push_back(x->Arg(1).Resolve().TryInstruction());
								sub_pending.push_back(x->Arg(0).Resolve().TryInstruction());
							} else if (xop == ValueOpcode::Phi) {
								for (size_t j = x->NumArgs(); j-- > 0;) {
									sub_pending.push_back(x->Arg(j).Resolve().TryInstruction());
								}
							}
						}
						for (const auto* s: sub) {
							if (implies_guard(Value(const_cast<Inst*>(g)),
							                  Value(const_cast<Inst*>(s)))) {
								cand.push_back(s);
							}
						}
						break;
					}
				}
			}
			std::vector<const Inst*> out;
			for (const auto* c: cand) {
				const bool redundant = std::ranges::any_of(cand, [&](const Inst* o) {
					return o != c && implies_guard(Value(const_cast<Inst*>(o)),
					                               Value(const_cast<Inst*>(c)));
				});
				if (!redundant && std::ranges::find(out, c) == out.end()) out.push_back(c);
			}
			std::ranges::sort(out, order_less);
			inv_cache.emplace(std::move(key), out);
			return out;
		};
		const auto invariant_neg_guards =
		    [&](std::span<const Inst* const> neg,
		        const Block* target_blk) -> std::vector<const Inst*> {
			std::vector<const Inst*> out;
			for (const auto* g: neg) {
				if (g != nullptr && strictly_dominates(g->Parent(), target_blk) &&
				    std::ranges::find(out, g) == out.end()) {
					out.push_back(g);
				}
			}
			std::ranges::sort(out, order_less);
			return out;
		};

		std::map<std::pair<GuardState, std::pair<const Inst*, const Inst*>>, GuardState>
		    expand_step_cache;
		const auto expand_guards = [&](GuardState state) -> GuardState {
			std::unordered_set<const Inst*> p_seen(state.pos.begin(), state.pos.end());
			std::unordered_set<const Inst*> n_seen(state.neg.begin(), state.neg.end());
			const auto add_pos = [&](Value v, bool& changed) {
				v = v.Resolve();
				if (v.IsImmediate()) {
					if (v.GetType() == Type::U1 && !v.U1()) state.contradictory = true;
					return;
				}
				if (const auto* i = v.TryInstruction(); i != nullptr && p_seen.insert(i).second) {
					order_of(i);
					state.pos.push_back(i);
					if (n_seen.contains(i)) state.contradictory = true;
					changed = true;
				}
			};
			const auto add_neg = [&](Value v, bool& changed) {
				v = v.Resolve();
				if (v.IsImmediate()) {
					if (v.GetType() == Type::U1 && v.U1()) state.contradictory = true;
					return;
				}
				if (const auto* i = v.TryInstruction(); i != nullptr && n_seen.insert(i).second) {
					order_of(i);
					state.neg.push_back(i);
					if (p_seen.contains(i)) state.contradictory = true;
					changed = true;
				}
			};
			bool changed = true;
			while (changed && !state.contradictory) {
				changed = false;
				for (size_t idx = 0; idx < state.pos.size() && !state.contradictory; ++idx) {
					const auto* g  = state.pos[idx];
					const auto  op = g->GetOpcode();
					if (op == ValueOpcode::LogicalAnd) {
						add_pos(g->Arg(0), changed);
						add_pos(g->Arg(1), changed);
					} else if (op == ValueOpcode::LogicalNot) {
						add_neg(g->Arg(0), changed);
					} else if (op == ValueOpcode::Phi) {
						for (const auto* c: invariant_guards(std::span(&g, 1u), g->Parent())) {
							add_pos(Value(const_cast<Inst*>(c)), changed);
						}
					}
				}
				for (size_t idx = 0; idx < state.neg.size() && !state.contradictory; ++idx) {
					const auto* g  = state.neg[idx];
					const auto  op = g->GetOpcode();
					if (op == ValueOpcode::LogicalOr) {
						add_neg(g->Arg(0), changed);
						add_neg(g->Arg(1), changed);
					} else if (op == ValueOpcode::LogicalNot) {
						add_pos(g->Arg(0), changed);
					} else if (op == ValueOpcode::LogicalAnd) {
						const auto pos_implies = [&](Value target) {
							for (size_t pi = 0; pi < state.pos.size(); ++pi) {
								if (implies_guard(Value(const_cast<Inst*>(state.pos[pi])), target))
									return true;
							}
							return false;
						};
						if (pos_implies(g->Arg(0))) add_neg(g->Arg(1), changed);
						if (pos_implies(g->Arg(1))) add_neg(g->Arg(0), changed);
					}
				}
			}
			std::ranges::sort(state.pos, order_less);
			std::ranges::sort(state.neg, order_less);
			return state;
		};
		const auto add_guard = [&](const GuardState& base, Value pos_add,
		                           Value neg_add) -> GuardState {
			if (base.contradictory) return base;
			pos_add = pos_add.Resolve();
			neg_add = neg_add.Resolve();
			const auto* pi = pos_add.TryInstruction();
			const auto* ni = neg_add.TryInstruction();
			if ((pos_add.IsImmediate() && pos_add.GetType() == Type::U1 && !pos_add.U1()) ||
			    (neg_add.IsImmediate() && neg_add.GetType() == Type::U1 && neg_add.U1())) {
				GuardState s;
				s.contradictory = true;
				return s;
			}
			const auto key = std::pair {base, std::pair {pi, ni}};
			if (const auto found = expand_step_cache.find(key); found != expand_step_cache.end()) {
				return found->second;
			}
			GuardState next = base;
			if (pi != nullptr) next.pos.push_back(pi);
			if (ni != nullptr) next.neg.push_back(ni);
			next = expand_guards(std::move(next));
			expand_step_cache.emplace(key, next);
			return next;
		};

		struct PhiKey {
			const Inst*              phi = nullptr;
			std::vector<const Inst*> pos;
			std::vector<const Inst*> neg;

			auto operator<=>(const PhiKey&) const = default;
		};
		std::map<PhiKey, std::optional<Range>>                             phi_ranges;
		std::vector<PhiKey>                                                phi_order;
		std::map<std::pair<const Inst*, GuardState>, std::optional<Range>> eval_memo;
		std::map<std::tuple<const Inst*, GuardState, bool>, bool>          pred_false_memo;
		std::set<std::pair<const Inst*, GuardState>>                       visiting;

		const auto pred_true = [&](Value cond, const GuardState& guards) {
			cond = cond.Resolve();
			if (cond.IsImmediate()) return cond.GetType() == Type::U1 && cond.U1();
			for (const auto* p: guards.pos) {
				if (implies_guard(Value(const_cast<Inst*>(p)), cond)) return true;
			}
			return false;
		};

		enum class TaskStep : uint8_t {
			EvalEnter,
			EvalRefinePos,
			EvalRefineNeg,
			EvalBinaryLeft,
			EvalBinaryRight,
			EvalShiftArg,
			EvalSelectPredFalse,
			EvalSelectTrueArm,
			EvalSelectFalseArm,
			PredEnter,
			PredAndLeft,
			PredAndRight,
			PredIEqRange,
			PredULtRange,
		};
		struct Task {
			TaskStep             step         = TaskStep::EvalEnter;
			Value                v;
			GuardState           guards;
			const Inst*          inst         = nullptr;
			bool                 optimistic   = false;
			bool                 in_visiting  = false;
			std::optional<Range> range_a;
			std::optional<Range> range_b;
			size_t               guard_idx    = 0u;
			uint8_t              sub_side     = 0u; // 0=lhs match, 1=rhs match
			bool                 tight_lt     = false;
			bool                 tight_strict = false;
			uint32_t             target_imm   = 0u;
		};

		const auto run_inductive = [&](bool start_pred_false, Value root_v,
		                               const GuardState& root_guards, bool root_opt,
		                               std::optional<Range>& out_range, bool& out_bool) {
			std::vector<Task> stack;
			stack.push_back(Task {
			    .step       = start_pred_false ? TaskStep::PredEnter : TaskStep::EvalEnter,
			    .v          = root_v,
			    .guards     = root_guards,
			    .optimistic = root_opt,
			});
			std::optional<Range> last_range;
			bool                 last_bool = false;

			const auto finish_eval = [&](Task& t, std::optional<Range> res) {
				if (t.in_visiting) {
					const auto mk = std::pair {t.inst, t.guards};
					visiting.erase(mk);
					eval_memo.emplace(mk, res);
				}
				last_range = res;
				stack.pop_back();
			};
			const auto finish_pred = [&](Task& t, bool res) {
				if (t.inst != nullptr && visiting.empty()) {
					pred_false_memo.emplace(std::tuple {t.inst, t.guards, t.optimistic}, res);
				}
				last_bool = res;
				stack.pop_back();
			};
			const auto push_eval = [&](Value v, const GuardState& g) {
				stack.push_back(Task {.step = TaskStep::EvalEnter, .v = v, .guards = g});
			};
			const auto push_pred = [&](Value cond, const GuardState& g, bool opt) {
				stack.push_back(
				    Task {.step = TaskStep::PredEnter, .v = cond, .guards = g, .optimistic = opt});
			};
			const auto start_refine = [&](Task& t, std::optional<Range> base_r) {
				if (!base_r || t.guards.contradictory) {
					finish_eval(t, t.guards.contradictory ? std::nullopt : base_r);
					return;
				}
				t.range_a   = base_r;
				t.guard_idx = 0u;
				t.sub_side  = 0u;
				t.step      = TaskStep::EvalRefinePos;
			};

			while (!stack.empty()) {
				auto& t = stack.back();
				switch (t.step) {
					case TaskStep::EvalEnter: {
						if (t.guards.contradictory) {
							finish_eval(t, std::nullopt);
							break;
						}
						t.v = t.v.Resolve();
						uint32_t imm = 0u;
						if (ImmediateU32(t.v, imm)) {
							start_refine(t, Range {imm, imm});
							break;
						}
						const auto* inst = t.v.TryInstruction();
						if (inst == nullptr) {
							start_refine(t, Range {});
							break;
						}
						t.inst        = inst;
						const auto mk = std::pair {inst, t.guards};
						if (const auto found = eval_memo.find(mk); found != eval_memo.end()) {
							finish_eval(t, found->second);
							break;
						}
						if (!visiting.insert(mk).second) {
							finish_eval(t, Range {});
							break;
						}
						t.in_visiting = true;

						switch (inst->GetOpcode()) {
							case ValueOpcode::Phi: {
								const auto   inv = invariant_guards(t.guards.pos, inst->Parent());
								const auto   inv_neg =
								    invariant_neg_guards(t.guards.neg, inst->Parent());
								const PhiKey key {inst, inv, inv_neg};
								const auto [it, inserted] =
								    phi_ranges.try_emplace(key, std::nullopt);
								if (inserted) phi_order.push_back(key);
								start_refine(t, it->second);
								break;
							}
							case ValueOpcode::ReadFirstLane: {
								if (inst->NumArgs() == 2u) {
									const auto mask = inst->Arg(1).Resolve();
									const bool implied =
									    mask == Value(true) ||
									    std::ranges::any_of(t.guards.pos, [&](const Inst* p) {
										    return implies_guard(Value(const_cast<Inst*>(p)), mask);
									    });
									if (implied) {
										t.step = TaskStep::EvalShiftArg; // pass through range
										push_eval(inst->Arg(0),
										          add_guard(GuardState {}, mask, Value {}));
										break;
									}
								}
								start_refine(t, Range {});
								break;
							}
							case ValueOpcode::SelectU32: {
								const auto cond = inst->Arg(0);
								if (pred_true(cond, t.guards)) {
									t.step = TaskStep::EvalShiftArg;
									push_eval(inst->Arg(1), t.guards);
									break;
								}
								t.step = TaskStep::EvalSelectPredFalse;
								push_pred(cond, t.guards, true);
								break;
							}
							case ValueOpcode::BitFieldUExtract: {
								uint32_t width = 0u;
								if (ImmediateU32(inst->Arg(2), width) && width < 32u) {
									start_refine(
									    t, Range {0u, width == 0u ? 0u : (1u << width) - 1u});
								} else {
									start_refine(t, Range {});
								}
								break;
							}
							case ValueOpcode::UMin32:
							case ValueOpcode::BitwiseAnd32:
							case ValueOpcode::IAdd32:
								t.step = TaskStep::EvalBinaryLeft;
								push_eval(inst->Arg(0), t.guards);
								break;
							case ValueOpcode::ShiftRightLogical32:
							case ValueOpcode::ShiftLeftLogical32: {
								uint32_t shift = 0u;
								if (ImmediateU32(inst->Arg(1), shift) && shift < 32u) {
									t.target_imm = shift;
									t.step       = TaskStep::EvalShiftArg;
									push_eval(inst->Arg(0), t.guards);
								} else {
									start_refine(t, Range {});
								}
								break;
							}
							default: start_refine(t, Range {}); break;
						}
						break;
					}
					case TaskStep::EvalSelectPredFalse: {
						const auto* inst = t.inst;
						if (last_bool) {
							t.step = TaskStep::EvalShiftArg;
							push_eval(inst->Arg(2), t.guards);
							break;
						}
						t.step = TaskStep::EvalSelectTrueArm;
						push_eval(inst->Arg(1), add_guard(t.guards, inst->Arg(0), Value {}));
						break;
					}
					case TaskStep::EvalSelectTrueArm: {
						t.range_a = last_range;
						t.step    = TaskStep::EvalSelectFalseArm;
						push_eval(t.inst->Arg(2), add_guard(t.guards, Value {}, t.inst->Arg(0)));
						break;
					}
					case TaskStep::EvalSelectFalseArm: {
						const auto& rt = t.range_a;
						const auto& rf = last_range;
						std::optional<Range> merged;
						if (!rt && !rf) merged = std::nullopt;
						else if (!rt) merged = rf;
						else if (!rf) merged = rt;
						else merged = Range {std::min(rt->low, rf->low), std::max(rt->high, rf->high)};
						start_refine(t, merged);
						break;
					}
					case TaskStep::EvalBinaryLeft:
						t.range_a = last_range;
						t.step    = TaskStep::EvalBinaryRight;
						push_eval(t.inst->Arg(1), t.guards);
						break;
					case TaskStep::EvalBinaryRight: {
						const auto& r0 = t.range_a;
						const auto& r1 = last_range;
						const auto  op = t.inst->GetOpcode();
						std::optional<Range> res = Range {};
						if (op == ValueOpcode::UMin32) {
							if (!r0 || !r1) res = std::nullopt;
							else res = Range {std::min(r0->low, r1->low), std::min(r0->high, r1->high)};
						} else if (op == ValueOpcode::BitwiseAnd32) {
							if (!r0 && !r1) res = std::nullopt;
							else if (!r0) res = Range {0u, r1->high};
							else if (!r1) res = Range {0u, r0->high};
							else res = Range {0u, std::min(r0->high, r1->high)};
						} else { // IAdd32
							if (!r0 || !r1) {
								res = std::nullopt;
							} else {
								const uint64_t lo_sum = uint64_t {r0->low} + r1->low;
								const uint64_t hi_sum = uint64_t {r0->high} + r1->high;
								constexpr uint64_t wrap = uint64_t {1} << 32u;
								if (hi_sum <= UINT32_MAX ||
								    (lo_sum >= wrap && hi_sum < (wrap << 1u))) {
									res = Range {static_cast<uint32_t>(lo_sum),
									             static_cast<uint32_t>(hi_sum)};
								}
							}
						}
						start_refine(t, res);
						break;
					}
					case TaskStep::EvalShiftArg: {
						const auto op = t.inst->GetOpcode();
						if (op == ValueOpcode::ReadFirstLane || op == ValueOpcode::SelectU32) {
							start_refine(t, last_range);
							break;
						}
						const auto& r0 = last_range;
						std::optional<Range> res = Range {};
						if (!r0) {
							res = std::nullopt;
						} else if (op == ValueOpcode::ShiftRightLogical32) {
							res = Range {r0->low >> t.target_imm, r0->high >> t.target_imm};
						} else if ((uint64_t {r0->high} << t.target_imm) <= UINT32_MAX) {
							res = Range {r0->low << t.target_imm, r0->high << t.target_imm};
						}
						start_refine(t, res);
						break;
					}
					case TaskStep::EvalRefinePos:
					case TaskStep::EvalRefineNeg: {
						if (t.sub_side != 0u) {
							// Returning from eval_range(bound, guards) triggered by previous guard check.
							const auto& br = last_range;
							if (br) {
								if (t.tight_lt) {
									if (t.tight_strict) {
										if (br->high == 0u) {
											finish_eval(t, std::nullopt);
											break;
										}
										t.range_a->high = std::min(t.range_a->high, br->high - 1u);
									} else {
										t.range_a->high = std::min(t.range_a->high, br->high);
									}
								} else {
									if (t.tight_strict) {
										if (br->low == UINT32_MAX) {
											finish_eval(t, std::nullopt);
											break;
										}
										t.range_a->low = std::max(t.range_a->low, br->low + 1u);
									} else {
										t.range_a->low = std::max(t.range_a->low, br->low);
									}
								}
							}
						}
						bool waiting_bound = false;
						if (t.step == TaskStep::EvalRefinePos) {
							while (t.guard_idx < t.guards.pos.size()) {
								const auto* g  = t.guards.pos[t.guard_idx];
								const auto  op = g->GetOpcode();
								if (op == ValueOpcode::ULessThan32 ||
								    op == ValueOpcode::ULessThanEqual32) {
									const bool strict = op == ValueOpcode::ULessThan32;
									if (t.sub_side == 0u) {
										t.sub_side = 1u;
										if (equiv(equiv, g->Arg(0), t.v)) {
											t.tight_lt     = true;
											t.tight_strict = strict;
											push_eval(g->Arg(1), t.guards);
											waiting_bound = true;
											break;
										}
									}
									if (t.sub_side == 1u) {
										t.sub_side = 2u;
										if (equiv(equiv, g->Arg(1), t.v)) {
											t.tight_lt     = false;
											t.tight_strict = strict;
											push_eval(g->Arg(0), t.guards);
											waiting_bound = true;
											break;
										}
									}
								} else if (op == ValueOpcode::UGreaterThan32 ||
								           op == ValueOpcode::UGreaterThanEqual32) {
									const bool strict = op == ValueOpcode::UGreaterThan32;
									if (t.sub_side == 0u) {
										t.sub_side = 1u;
										if (equiv(equiv, g->Arg(1), t.v)) {
											t.tight_lt     = true;
											t.tight_strict = strict;
											push_eval(g->Arg(0), t.guards);
											waiting_bound = true;
											break;
										}
									}
									if (t.sub_side == 1u) {
										t.sub_side = 2u;
										if (equiv(equiv, g->Arg(0), t.v)) {
											t.tight_lt     = false;
											t.tight_strict = strict;
											push_eval(g->Arg(1), t.guards);
											waiting_bound = true;
											break;
										}
									}
								} else if (op == ValueOpcode::INotEqual32) {
									if ((g->Arg(0).Resolve() == Value(0u) &&
									     equiv(equiv, g->Arg(1), t.v)) ||
									    (g->Arg(1).Resolve() == Value(0u) &&
									     equiv(equiv, g->Arg(0), t.v))) {
										t.range_a->low = std::max(t.range_a->low, 1u);
									}
								}
								++t.guard_idx;
								t.sub_side = 0u;
							}
							if (waiting_bound) break;
							t.step      = TaskStep::EvalRefineNeg;
							t.guard_idx = 0u;
							t.sub_side  = 0u;
						}
						while (t.guard_idx < t.guards.neg.size()) {
							const auto* g  = t.guards.neg[t.guard_idx];
							const auto  op = g->GetOpcode();
							if (op == ValueOpcode::UGreaterThanEqual32 ||
							    op == ValueOpcode::UGreaterThan32) {
								const bool strict = op == ValueOpcode::UGreaterThanEqual32;
								if (t.sub_side == 0u) {
									t.sub_side = 1u;
									if (equiv(equiv, g->Arg(0), t.v)) {
										t.tight_lt     = true;
										t.tight_strict = strict;
										push_eval(g->Arg(1), t.guards);
										waiting_bound = true;
										break;
									}
								}
								if (t.sub_side == 1u) {
									t.sub_side = 2u;
									if (equiv(equiv, g->Arg(1), t.v)) {
										t.tight_lt     = false;
										t.tight_strict = strict;
										push_eval(g->Arg(0), t.guards);
										waiting_bound = true;
										break;
									}
								}
							} else if (op == ValueOpcode::ULessThanEqual32 ||
							           op == ValueOpcode::ULessThan32) {
								const bool strict = op == ValueOpcode::ULessThanEqual32;
								if (t.sub_side == 0u) {
									t.sub_side = 1u;
									if (equiv(equiv, g->Arg(1), t.v)) {
										t.tight_lt     = true;
										t.tight_strict = strict;
										push_eval(g->Arg(0), t.guards);
										waiting_bound = true;
										break;
									}
								}
								if (t.sub_side == 1u) {
									t.sub_side = 2u;
									if (equiv(equiv, g->Arg(0), t.v)) {
										t.tight_lt     = false;
										t.tight_strict = strict;
										push_eval(g->Arg(1), t.guards);
										waiting_bound = true;
										break;
									}
								}
							} else if (op == ValueOpcode::IEqual32) {
								if ((g->Arg(0).Resolve() == Value(0u) &&
								     equiv(equiv, g->Arg(1), t.v)) ||
								    (g->Arg(1).Resolve() == Value(0u) &&
								     equiv(equiv, g->Arg(0), t.v))) {
									t.range_a->low = std::max(t.range_a->low, 1u);
								}
							}
							++t.guard_idx;
							t.sub_side = 0u;
						}
						if (waiting_bound) break;
						if (t.range_a->low > t.range_a->high) finish_eval(t, std::nullopt);
						else finish_eval(t, t.range_a);
						break;
					}
					case TaskStep::PredEnter: {
						if (t.guards.contradictory) {
							finish_pred(t, t.optimistic);
							break;
						}
						const auto cond = t.v.Resolve();
						if (cond.IsImmediate()) {
							finish_pred(t, cond.GetType() == Type::U1 && !cond.U1());
							break;
						}
						bool negated_in_guard = false;
						for (const auto* n: t.guards.neg) {
							if (equiv(equiv, cond, Value(const_cast<Inst*>(n)))) {
								negated_in_guard = true;
								break;
							}
						}
						if (negated_in_guard) {
							finish_pred(t, true);
							break;
						}
						const auto* inst = cond.TryInstruction();
						if (inst == nullptr) {
							finish_pred(t, false);
							break;
						}
						if (const auto found =
						        pred_false_memo.find({inst, t.guards, t.optimistic});
						    found != pred_false_memo.end()) {
							last_bool = found->second;
							stack.pop_back();
							break;
						}
						t.inst        = inst;
						const auto op = inst->GetOpcode();
						if (op == ValueOpcode::LogicalAnd) {
							t.step = TaskStep::PredAndLeft;
							push_pred(inst->Arg(0), add_guard(t.guards, inst->Arg(1), Value {}),
							          t.optimistic);
							break;
						}
						if (op == ValueOpcode::IEqual32) {
							bool matched = false;
							for (uint32_t i = 0; i < 2u; ++i) {
								uint32_t target = 0u;
								if (ImmediateU32(inst->Arg(i), target)) {
									t.target_imm = target;
									t.step       = TaskStep::PredIEqRange;
									push_eval(inst->Arg(i ^ 1u), t.guards);
									matched = true;
									break;
								}
							}
							if (matched) break;
						}
						uint32_t target = 0u;
						if (op == ValueOpcode::ULessThan32 && ImmediateU32(inst->Arg(0), target)) {
							t.target_imm = target;
							t.step       = TaskStep::PredULtRange;
							push_eval(inst->Arg(1), t.guards);
							break;
						}
						finish_pred(t, false);
						break;
					}
					case TaskStep::PredAndLeft:
						if (last_bool) {
							finish_pred(t, true);
							break;
						}
						t.step = TaskStep::PredAndRight;
						push_pred(t.inst->Arg(1), add_guard(t.guards, t.inst->Arg(0), Value {}),
						          t.optimistic);
						break;
					case TaskStep::PredAndRight: finish_pred(t, last_bool); break;
					case TaskStep::PredIEqRange:
						if (!last_range) finish_pred(t, t.optimistic);
						else finish_pred(t, t.target_imm < last_range->low ||
						                        t.target_imm > last_range->high);
						break;
					case TaskStep::PredULtRange:
						if (!last_range) finish_pred(t, t.optimistic);
						else finish_pred(t, last_range->high <= t.target_imm);
						break;
				}
			}
			out_range = last_range;
			out_bool  = last_bool;
		};

		const auto pred_false = [&](Value cond, const GuardState& guards, bool optimistic) {
			std::optional<Range> dummy_range;
			bool                 ans = false;
			run_inductive(true, cond, guards, optimistic, dummy_range, ans);
			return ans;
		};
		const auto eval_range = [&](Value v, const GuardState& guards) {
			std::optional<Range> ans;
			bool                 dummy_bool = false;
			run_inductive(false, v, guards, false, ans, dummy_bool);
			return ans;
		};

		const GuardState empty_guards;
		for (auto* select: candidate_selects) {
			pred_false(select->Arg(0), empty_guards, true);
		}

		std::map<PhiKey, std::pair<uint32_t, uint32_t>> widen_counts;
		bool                                            changed = true;
		while (changed) {
			eval_memo.clear();
			pred_false_memo.clear();
			changed = false;
			for (size_t idx = 0; idx < phi_order.size(); ++idx) {
				const auto  key = phi_order[idx];
				const auto* phi = key.phi;
				GuardState  guards;
				guards.pos = key.pos;
				guards.neg = key.neg;
				guards     = expand_guards(std::move(guards));
				std::optional<Range> joined;
				if (!guards.contradictory) {
					const bool has_blocks = phi->NumPhiBlocks() == phi->NumArgs();
					for (size_t arm = 0; arm < phi->NumArgs(); ++arm) {
						GuardState arm_guards = guards;
						if (has_blocks) {
							const Block* pred = phi->PhiBlock(arm);
							if (IsUnreachablePhiEdge(pred, phi->Parent())) continue;
							const Block* to = phi->Parent();
							for (uint32_t hop = 0u; hop < 8u && pred != nullptr; ++hop) {
								if (const auto edge = ConditionalEdge(pred, to);
								    edge && edge->lanes == LaneQuantifier::All) {
									const auto pos_add =
									    edge->positive ? edge->condition : Value {};
									const auto neg_add =
									    edge->positive ? Value {} : edge->condition;
									arm_guards = add_guard(arm_guards, pos_add, neg_add);
									if (arm_guards.contradictory) break;
								}
								if (pred->ImmPredecessors().size() != 1u) break;
								to   = pred;
								pred = pred->ImmPredecessors()[0];
							}
							if (arm_guards.contradictory) continue;
						}
						const auto arm_range = eval_range(phi->Arg(arm), arm_guards);
						if (!arm_range) continue;
						if (!joined) joined = arm_range;
						else {
							joined->low  = std::min(joined->low, arm_range->low);
							joined->high = std::max(joined->high, arm_range->high);
						}
					}
				}
				auto& slot = phi_ranges.at(key);
				if (joined && slot) {
					auto& [lo_widen, hi_widen] = widen_counts[key];
					if (joined->low < slot->low && ++lo_widen > 32u) joined->low = 0u;
					if (joined->high > slot->high && ++hi_widen > 32u) joined->high = UINT32_MAX;
				}
				if (joined != slot) {
					slot    = joined;
					changed = true;
				}
			}
		}

		eval_memo.clear();
		pred_false_memo.clear();
		bool folded_any = false;
		for (auto* select: candidate_selects) {
			if (pred_false(select->Arg(0), empty_guards, false)) {
				select->ReplaceUsesWith(select->Arg(2).Resolve());
				folded_any = true;
			}
		}
		if (!folded_any) return;

		bool phi_changed = true;
		while (phi_changed) {
			phi_changed = false;
			for (auto* block: m_program.blocks) {
				for (auto& inst: *block) {
					if (inst.GetOpcode() != ValueOpcode::Phi || !inst.HasUses()) continue;
					Value      same;
					bool       uniform    = true;
					const bool has_blocks = inst.NumPhiBlocks() == inst.NumArgs();
					for (size_t arm = 0; arm < inst.NumArgs(); ++arm) {
						if (has_blocks && IsUnreachablePhiEdge(inst.PhiBlock(arm), inst.Parent())) {
							continue;
						}
						const auto arg = inst.Arg(arm).Resolve();
						if (arg.TryInstruction() == &inst) continue;
						if (same.IsEmpty()) same = arg;
						else if (same != arg) {
							uniform = false;
							break;
						}
					}
					if (uniform && !same.IsEmpty()) {
						inst.ReplaceUsesWith(same);
						phi_changed = true;
					}
				}
			}
		}
	}

	bool BoundedSelectedIndex(Value index, Value active,
	                          DescriptorSource::IndirectDescriptor& indirect) const {
		const auto* selected = index.Resolve().TryInstruction();
		if (selected == nullptr || selected->GetOpcode() != ValueOpcode::Phi ||
		    selected->NumArgs() == 0u) return false;
		std::vector<const Inst*> values {selected};
		for (size_t i = 0; i < values.size(); ++i) {
			const auto* value = values[i];
			if (value->GetOpcode() != ValueOpcode::Phi &&
			    value->GetOpcode() != ValueOpcode::SelectU32) continue;
			for (size_t arm = value->GetOpcode() == ValueOpcode::SelectU32 ? 1u : 0u;
			     arm < value->NumArgs(); ++arm) {
				const auto* next = value->Arg(arm).Resolve().TryInstruction();
				if (next != nullptr && std::ranges::find(values, next) == values.end())
					values.push_back(next);
			}
		}
		for (const auto* candidate: values) {
			if (candidate->GetOpcode() != ValueOpcode::IAdd32) continue;
			for (uint32_t arg = 0; arg < 2u; ++arg) {
				const auto* bound = BoundedLoop(candidate->Arg(arg), candidate->Parent());
				const auto first = candidate->Arg(arg ^ 1u);
				if (bound == nullptr || !ValidateRuntimeValue(m_program, first, RuntimeValueType::Integer))
					continue;
				const auto* induction = candidate->Arg(arg).ResolveInstruction();
				const Inst* carried = nullptr;
				for (const auto* value: values) {
					if (value->GetOpcode() == ValueOpcode::Phi && value->Parent() == induction->Parent()) {
						if (carried != nullptr) { carried = nullptr; break; }
						carried = value;
					}
				}
				if (carried == nullptr || carried->NumArgs() != 2u ||
				    carried->PhiBlock(0) != induction->PhiBlock(0) ||
				    carried->PhiBlock(1) != induction->PhiBlock(1)) continue;
				std::vector<std::pair<Value, bool>> exits;
				for (size_t arm = 0; arm < selected->NumArgs(); ++arm) {
					const auto edge = ConditionalEdge(selected->PhiBlock(arm), selected->Parent());
					if (!edge || edge->lanes != LaneQuantifier::All) break;
					exits.emplace_back(edge->condition, edge->positive);
				}
				if (exits.size() != selected->NumArgs()) continue;
				for (const auto& mask: *induction->Parent()) {
					if (mask.GetOpcode() != ValueOpcode::Phi || mask.GetType() != Type::U1 ||
					    mask.NumArgs() != 2u || mask.PhiBlock(0) != carried->PhiBlock(0) ||
					    mask.PhiBlock(1) != carried->PhiBlock(1)) continue;
					const auto initial = induction->Arg(0).Resolve().IsImmediate() ? 0u : 1u;
					const auto enabled = mask.Arg(initial).Resolve();
					if (!enabled.IsImmediate() || enabled.GetType() != Type::U1 || !enabled.U1()) continue;
					LoopBoundProof proof(m_program, *induction, *bound);
					if (!proof.SelectsRange(*selected, active, *candidate, *carried, mask, exits)) continue;
					indirect.selector_first = first;
					indirect.key_count = bound->Arg(1);
					indirect.signed_count = bound->GetOpcode() == ValueOpcode::SLessThan32;
					return true;
				}
			}
		}
		return false;
	}

	using SelectorBounds = std::span<const std::pair<const Inst*, uint32_t>>;

	// Each equation is an inductive subset obligation. Phis require every incoming
	// value; AND needs either operand. Grounding rejects cycles without an initial
	// value, then the greatest fixed point checks every initial and carried edge.
	bool BitsetMaskSubset(Value mask, Value active, Value word,
	                      std::span<const Inst* const> reads) const {
		struct State {
			Value value;
			Value guard;
			uint32_t word = UINT32_MAX;
			uint32_t component = UINT32_MAX; // U1 implication of the load guard.
			bool operator==(const State&) const = default;
		};
		struct Equation {
			bool all = true;
			std::vector<size_t> terms;
		};
		// A scalar load reads one word without EXEC; four-word loads take their guard.
		const bool scalar = reads[0]->GetOpcode() == ValueOpcode::ReadConstBuffer;
		const auto load_guard = scalar ? Value(true) : reads[0]->Arg(4).Resolve();
		const auto words = static_cast<uint32_t>(reads.size()) * (scalar ? 1u : 4u);
		std::vector<State> states(2);
		std::vector<Equation> equations(2);
		const auto add = [&](State state) {
			state.value = state.value.Resolve();
			const auto found = std::find(states.begin() + 2u, states.end(), state);
			if (found != states.end()) return static_cast<size_t>(found - states.begin());
			states.push_back(state);
			equations.emplace_back();
			return states.size() - 1u;
		};
		std::vector<size_t> roots {add({.value = active})};
		for (uint32_t component = 0; component < words; ++component)
			roots.push_back(add({mask, active, component, component}));
		for (size_t index = 2; index < states.size(); ++index) {
			const auto state = states[index];
			const auto* inst = state.value.TryInstruction();
			Equation equation;
			const auto term = [&](Value value, Value guard, uint32_t selected) {
				return add({value, guard, selected, state.component});
			};
			if (state.component == UINT32_MAX) {
				if (Implies(state.value, load_guard)) equation.terms = {1u};
				else if (inst != nullptr && (inst->GetOpcode() == ValueOpcode::LogicalAnd ||
				                            inst->GetOpcode() == ValueOpcode::Phi)) {
					equation.all = inst->GetOpcode() == ValueOpcode::Phi;
					for (size_t arg = 0; arg < inst->NumArgs(); ++arg)
						equation.terms.push_back(add({.value = inst->Arg(arg)}));
				}
			} else if (state.value == Value(0u)) {
				equation.terms = {1u};
			} else if (inst != nullptr) {
				const auto constant = [&](const auto& self, Value value) -> std::optional<uint32_t> {
					uint32_t literal;
					if (ImmediateU32(value, literal)) return literal;
					if (state.word != UINT32_MAX && EquivalentValue(m_program, value, word)) return state.word;
					const auto* operation = value.Resolve().TryInstruction();
					if (operation == nullptr || operation->GetOpcode() != ValueOpcode::BitwiseAnd32) return {};
					const auto left = self(self, operation->Arg(0));
					const auto right = self(self, operation->Arg(1));
					return left && right ? std::optional {*left & *right} : std::nullopt;
				};
				const auto condition = [&](const auto& self, Value value) -> std::optional<bool> {
					value = value.Resolve();
					if (value.IsImmediate() && value.GetType() == Type::U1) return value.U1();
					if (Implies(state.guard, value) || Implies(load_guard, value)) return true;
					const auto* test = value.TryInstruction();
					if (test == nullptr) return {};
					if (test->GetOpcode() == ValueOpcode::LogicalAnd) {
						const auto left = self(self, test->Arg(0));
						const auto right = self(self, test->Arg(1));
						if ((left && !*left) || (right && !*right)) return false;
						if (left && right) return *left && *right;
					} else if (test->GetOpcode() == ValueOpcode::IEqual32) {
						const auto left = constant(constant, test->Arg(0));
						const auto right = constant(constant, test->Arg(1));
						if (left && right) return *left == *right;
					}
					return {};
				};
				switch (inst->GetOpcode()) {
					case ValueOpcode::CompositeExtractU32x4: {
						uint32_t component;
						if (inst->Arg(0).Resolve().TryInstruction() == reads[state.component / 4u] &&
						    ImmediateU32(inst->Arg(1), component) && component == state.component % 4u)
							equation.terms = {1u};
						break;
					}
					case ValueOpcode::ReadConstBuffer:
						if (scalar && inst == reads[state.component]) equation.terms = {1u};
						break;
					case ValueOpcode::Phi:
						// The predecessor may be an earlier outer iteration. Establish
						// this register's invariant without the current word or EXEC.
						for (size_t arg = 0; arg < inst->NumArgs(); ++arg)
							equation.terms.push_back(term(inst->Arg(arg), load_guard, UINT32_MAX));
						break;
					case ValueOpcode::BitwiseAnd32:
						equation.all = false;
						equation.terms = {term(inst->Arg(0), state.guard, state.word),
						                  term(inst->Arg(1), state.guard, state.word)};
						break;
					case ValueOpcode::SelectU32: {
						const auto selected = condition(condition, inst->Arg(0));
						if (!selected || *selected) {
							auto selected_word = state.word;
							const auto equality = EqualLocalValue(inst->Arg(0), [&](Value key) {
								key = key.Resolve();
								const auto* masked = key.TryInstruction();
								uint32_t bound = 0;
								if (masked != nullptr && masked->GetOpcode() == ValueOpcode::BitwiseAnd32 &&
								    DescendingLoop(word, masked->Parent(), bound) && bound == words - 1u) {
									for (uint32_t side = 0; side < 2u; ++side) {
										uint32_t bits;
										if (ImmediateU32(masked->Arg(side), bits) && (bits & bound) == bound)
											key = masked->Arg(side ^ 1u).Resolve();
									}
								}
								return EquivalentValue(m_program, key, word);
							});
							uint32_t literal;
							if (ImmediateU32(equality, literal) && literal < words) selected_word = literal;
							equation.terms.push_back(term(inst->Arg(1), inst->Arg(0), selected_word));
						}
						if (!selected || !*selected)
							equation.terms.push_back(term(inst->Arg(2), state.guard, state.word));
						break;
					}
					default: break;
				}
			}
			if (equation.terms.empty()) equation.terms = {0u};
			equations[index] = std::move(equation);
		}
		std::vector<bool> valid(states.size(), false);
		valid[1] = true;
		bool changed;
		do {
			changed = false;
			for (size_t index = 2; index < equations.size(); ++index) {
				if (!valid[index] && std::ranges::any_of(equations[index].terms,
				    [&](size_t term) { return valid[term]; })) valid[index] = changed = true;
			}
		} while (changed);
		do {
			changed = false;
			for (size_t index = 2; index < equations.size(); ++index) {
				const auto& equation = equations[index];
				const auto proven = [&](size_t term) { return valid[term]; };
				if (valid[index] && !(equation.all ? std::ranges::all_of(equation.terms, proven)
				                                  : std::ranges::any_of(equation.terms, proven))) {
					valid[index] = false;
					changed = true;
				}
			}
		} while (changed);
		return std::ranges::all_of(roots, [&](size_t root) { return valid[root]; });
	}

	// The mask words whose set bits are every record a bit scan of index selects: the scanned
	// value keeps only bits of one scalar load, which reads a word its address can reach.
	std::optional<DescriptorSource::IndirectDescriptor::SelectorRead::MaskRanks>
	BitScanRanks(Value index) {
		const auto* scan = index.Resolve().TryInstruction();
		if (scan == nullptr || scan->GetOpcode() != ValueOpcode::FindILsb32) return {};
		const auto* masked = scan->Arg(0).Resolve().TryInstruction();
		if (masked == nullptr) return {};
		std::vector<const Inst*> values {masked};
		const Inst* load = nullptr;
		for (size_t i = 0; i < values.size() && load == nullptr; ++i) {
			const auto* value = values[i];
			if (value->GetOpcode() == ValueOpcode::ReadConstBuffer) load = value;
			if (value->GetOpcode() != ValueOpcode::Phi &&
			    value->GetOpcode() != ValueOpcode::BitwiseAnd32) continue;
			for (size_t arg = 0; arg < value->NumArgs(); ++arg) {
				const auto* next = value->Arg(arg).Resolve().TryInstruction();
				if (next != nullptr && std::ranges::find(values, next) == values.end())
					values.push_back(next);
			}
		}
		uint32_t memory_index = 0;
		const auto* memory = load != nullptr ? ScalarReadMemory(*load, memory_index) : nullptr;
		if (memory == nullptr || memory->offset > INT32_MAX ||
		    !MemoryIndexBelongsTo(memory_index, *load) ||
		    !BitsetMaskSubset(scan->Arg(0), Value(true), Value(0u), std::array {load})) return {};
		auto address = load->Arg(1);
		auto active = Value(true);
		const auto* lane = address.Resolve().TryInstruction();
		if (lane != nullptr && lane->GetOpcode() == ValueOpcode::ReadFirstLane) {
			if (!HasActiveLane(lane->Arg(1), lane->Parent())) return {};
			active = lane->Arg(1);
			address = lane->Arg(0);
		}
		// A left shift keeps the address a multiple of its scale; other addresses read any word.
		const auto* shift = ActiveValue(address, active).TryInstruction();
		uint32_t bits = 0;
		if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
		    !ImmediateU32(shift->Arg(1), bits) || bits >= 32u) bits = 0;
		DescriptorSource source;
		if (!MakeRuntimeTableSource(*load, source)) return {};
		return DescriptorSource::IndirectDescriptor::SelectorRead::MaskRanks {
		    .source = InternSource(source), .stride = std::max(1u << bits, 4u),
		    .offset = memory->offset & ~3u};
	}

	bool MatchBitsetSelector(Value rank, Value active, SelectorBounds bounds, uint32_t words,
	                         DescriptorSource::IndirectDescriptor& indirect) {
		const auto selected_value = [&](Value value) {
			for (;;) {
				value = ActiveValue(value, active);
				const auto* select = value.TryInstruction();
				if (select == nullptr || select->GetOpcode() != ValueOpcode::SelectU32 ||
				    !SelectorPredicateFalse(select->Arg(0), bounds, 0u, active)) return value;
				value = select->Arg(2);
			}
		};
		const auto* sum = selected_value(rank).TryInstruction();
		if (sum == nullptr || sum->GetOpcode() != ValueOpcode::IAdd32) return false;
		Value word;
		Value bit;
		for (uint32_t arg = 0; arg < 2u; ++arg) {
			const auto* shift = selected_value(sum->Arg(arg)).TryInstruction();
			if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
			    shift->Arg(1).Resolve() != Value(5u)) continue;
			word = selected_value(shift->Arg(0));
			bit = selected_value(sum->Arg(arg ^ 1u));
		}
		const auto* word_inst = word.TryInstruction();
		if (word_inst == nullptr || std::ranges::none_of(bounds, [&](const auto& bound) {
			return bound.first == word_inst && bound.second == words - 1u;
		})) return false;
		// FFBH's leading-zero count is inverted again to recover the set-bit rank.
		for (uint32_t inverse = 0; inverse < 2u; ++inverse) {
			const auto* subtract = bit.TryInstruction();
			if (subtract == nullptr || subtract->GetOpcode() != ValueOpcode::ISub32 ||
			    subtract->Arg(0).Resolve() != Value(31u)) return false;
			bit = selected_value(subtract->Arg(1));
		}
		const auto* scan = bit.TryInstruction();
		if (scan == nullptr || scan->GetOpcode() != ValueOpcode::FindUMsb32 ||
		    !ImpliesNonzero(active, scan->Arg(0))) return false;
		const auto bitmap_load = [&](const Inst& read, uint32_t offset) {
			const auto index = read.Flags<MemoryFlags>().index;
			if (read.GetOpcode() != ValueOpcode::LoadBufferU32x4 || InCycle(read.Parent()) ||
			    index >= m_program.memory_info.size()) return false;
			const auto& memory = m_program.memory_info[index];
			return memory.kind == ResourceKind::Buffer && !memory.typed && !memory.formatted &&
			       memory.idxen && !memory.offen && memory.offset == offset &&
			       memory.data_bits == 32u && memory.data_dwords == 4u &&
			       read.Arg(2).Resolve() == Value(0u) && read.Arg(3).Resolve() == Value(0u);
		};
		for (const auto* block: m_program.blocks) {
			for (const auto& read: *block) {
				if (!bitmap_load(read, 0u)) continue;
				// An eight-word record continues in a second four-word load of the same buffer and EXEC.
				std::array<const Inst*, 2> reads {&read};
				if (words == 8u)
					for (const auto* other: m_program.blocks)
						for (const auto& load: *other)
							if (reads[1] == nullptr && bitmap_load(load, 16u) &&
							    EquivalentValue(m_program, load.Arg(0), read.Arg(0)) &&
							    EquivalentValue(m_program, load.Arg(4), read.Arg(4))) reads[1] = &load;
				const auto parts = std::span(reads).first(words / 4u);
				DescriptorSource source;
				if (std::ranges::find(parts, nullptr) != parts.end() ||
				    !BitsetMaskSubset(scan->Arg(0), active, word, parts) ||
				    !MakeRuntimeTableSource(read, source)) continue;
				indirect.selector.emplace(DescriptorSource::IndirectDescriptor::SelectorRead {
				    .source = InternSource(source), .stride = words * 4u, .offset = 0u, .bitset = true});
				return true;
			}
		}
		return false;
	}

	bool MatchReversedImageRange(Value key, DescriptorSource::IndirectDescriptor& indirect) {
		const auto* subtract = key.Resolve().TryInstruction();
		if (subtract == nullptr || subtract->GetOpcode() != ValueOpcode::ISub32 ||
		    !ValidateRuntimeValue(m_program, subtract->Arg(0), RuntimeValueType::Integer)) return false;
		const auto* lane = subtract->Arg(1).Resolve().TryInstruction();
		if (lane == nullptr || lane->GetOpcode() != ValueOpcode::ReadFirstLane ||
		    !HasActiveLane(lane->Arg(1), lane->Parent())) return false;
		std::vector<std::pair<const Inst*, uint32_t>> bounds;
		for (const auto* block: m_program.blocks) {
			for (const auto& inst: *block) {
				if (inst.GetOpcode() != ValueOpcode::Phi) continue;
				uint32_t maximum = 0;
				if (DescendingLoop(Value(const_cast<Inst*>(&inst)), lane->Parent(), maximum))
					bounds.emplace_back(&inst, maximum);
			}
		}
		const auto maximum = SelectorMaximum(lane->Arg(0), bounds, 0u, lane->Arg(1));
		if (maximum >= INT32_MAX) return false;
		auto& first = m_program.value_storage.emplace_back(ValueOpcode::ISub32);
		first.SetArg(0, subtract->Arg(0));
		first.SetArg(1, Value(maximum));
		indirect.selector_first = Value(&first);
		indirect.key_count = Value(maximum + 1u);
		if (maximum == 127u || maximum == 255u)
			MatchBitsetSelector(lane->Arg(0), lane->Arg(1), bounds, (maximum + 1u) / 32u, indirect);
		return true;
	}

	bool IndirectBufferAccess(const Inst& inst) const {
		const auto opcode = inst.GetOpcode();
		if (opcode != ValueOpcode::StoreBufferU32 && opcode != ValueOpcode::StoreBufferU32x2 &&
		    opcode != ValueOpcode::StoreBufferU8 &&
		    opcode != ValueOpcode::StoreBufferU32x4 && opcode != ValueOpcode::LoadBufferU32x2 &&
		    opcode != ValueOpcode::LoadBufferU32x3 && opcode != ValueOpcode::LoadBufferU32x4) return false;
		const auto index = inst.Flags<MemoryFlags>().index;
		if (index >= m_program.memory_info.size()) return false;
		const auto& memory = m_program.memory_info[index];
		if (memory.kind != ResourceKind::Buffer || memory.typed || (memory.idxen && memory.offen) ||
		    (memory.idxen && inst.Arg(2).Resolve() != Value(0u)) ||
		    inst.Arg(3).Resolve() != Value(0u)) return false;
		if (memory.idxen && memory.offset != 0u &&
		    (memory.formatted || memory.data_bits != 32u || (memory.offset & 3u) != 0u)) return false;
		if (opcode == ValueOpcode::StoreBufferU32)
			return !memory.formatted || (memory.idxen && memory.data_bits == 32u && memory.data_dwords == 1u);
		if (!memory.idxen) return false;
		if (opcode == ValueOpcode::StoreBufferU8)
			return !memory.formatted && memory.data_bits == 8u;
		const auto components = BufferComponentCount(opcode);
		if (opcode == ValueOpcode::StoreBufferU32x2 || opcode == ValueOpcode::StoreBufferU32x4)
			return memory.data_bits == 32u && memory.data_dwords == components &&
			       (!memory.formatted || (components == 4u && memory.component_count == 4u));
		return memory.formatted && memory.data_dwords == components &&
		       ((memory.data_bits == 32u && memory.component_count == components) ||
		        (opcode == ValueOpcode::LoadBufferU32x2 && memory.data_bits == 16u &&
		         memory.component_count == 4u));
	}

	bool MatchDescriptorReads(Inst& handle, const DescriptorSource& descriptor,
	                          IndirectDescriptorPlan& plan, Value& key, uint32_t& table_offset,
	                          uint32_t& table_stride, uint32_t& table_immediate,
	                          bool allow_fixed = false) {
		Inst* table_handle = nullptr;
		Value table_guard;
		const uint32_t alignment_mask = allow_fixed ? ~3u : UINT32_MAX;
		for (uint32_t dword = 0; dword < plan.read_dwords; ++dword) {
			auto* read = descriptor.dwords[dword].Resolve().TryInstruction();
			if (read == nullptr) {
				return false;
			}
			uint32_t memory_index = 0;
			const auto* memory = ScalarReadMemory(*read, memory_index);
			uint32_t component = 0;
			const bool indexed = memory == nullptr;
			if (indexed) {
				if (handle.NumArgs() != 8u) return false;
				const auto* selected = UniformizedMaterialValue(descriptor.dwords[dword], handle);
				if (selected == nullptr) return false;
				const auto* extract = selected->Arg(1).Resolve().TryInstruction();
				if (extract == nullptr || extract->GetOpcode() != ValueOpcode::CompositeExtractU32x4 ||
				    !ImmediateU32(extract->Arg(1), component) || component >= 4u) return false;
				read = extract->Arg(0).Resolve().TryInstruction();
				if (read == nullptr || read->GetOpcode() != ValueOpcode::LoadBufferU32x4) return false;
				memory_index = read->Flags<MemoryFlags>().index;
				if (memory_index >= m_program.memory_info.size()) return false;
				memory = &m_program.memory_info[memory_index];
				uint32_t offset = 1u, scalar = 1u;
				if (memory->kind != ResourceKind::Buffer || memory->typed || memory->formatted ||
				    !memory->idxen || memory->offen || memory->data_bits != 32u ||
				    memory->data_dwords != 4u ||
				    !ImmediateU32(read->Arg(2), offset) || offset != 0u ||
				    !ImmediateU32(read->Arg(3), scalar) || scalar != 0u ||
				    !EquivalentValue(m_program, read->Arg(4), selected->Arg(0)) ||
				    (dword != 0u && !EquivalentValue(m_program, table_guard, read->Arg(4)))) return false;
				table_guard = read->Arg(4);
			}
			if (memory->offset > INT32_MAX || (!allow_fixed && (memory->offset & 3u) != 0u) ||
			    !MemoryIndexBelongsTo(memory_index, *read)) {
				return false;
			}
			auto* current_handle = read->Arg(0).Resolve().TryInstruction();
			Value current_key;
			uint32_t offset = 0;
			uint32_t stride = 0;
			if (current_handle == nullptr ||
			    current_handle->GetOpcode() != (memory->kind == ResourceKind::ScalarAddress
			                                      ? ValueOpcode::GetAddressResource
			                                      : ValueOpcode::GetBufferResource) ||
			    (memory->kind == ResourceKind::ScalarAddress && !AvailableBefore(*read, handle)) ||
			    (table_handle != nullptr &&
			     !EquivalentValue(m_program, Value(table_handle), Value(current_handle))) ||
			    (!indexed && !(allow_fixed && ImmediateU32(read->Arg(1), offset)) &&
			     !MatchTableOffset(read->Arg(1), current_key, offset, stride))) {
				return false;
			}
			if (indexed) {
				current_key = read->Arg(1);
				offset = component * 4u;
			}
			if (dword == 0u) {
				if (component != 0u) return false;
				plan.table_indexed = indexed;
				key = current_key;
				table_offset = offset;
				table_immediate = memory->offset;
				table_stride = stride;
			} else if (plan.table_indexed != indexed || table_stride != stride ||
			           !EquivalentValue(m_program, key, current_key) ||
			           uint64_t {table_offset & alignment_mask} + (table_immediate & alignment_mask) +
			                   dword * sizeof(uint32_t) !=
			               uint64_t {offset & alignment_mask} + (memory->offset & alignment_mask)) {
				return false;
			}
			plan.split_offsets |= !indexed && offset != table_offset;
			table_handle = current_handle;
			if (handle.GetOpcode() != ValueOpcode::GetBufferResource) {
				plan.retain_reads |= handle.GetOpcode() == ValueOpcode::GetSamplerResource ||
				    (memory->kind == ResourceKind::ScalarAddress && !UsesOnlyImageDescriptors(*read));
			} else if (std::ranges::any_of(read->Uses(), [&](const Use& use) {
				return use.user->GetOpcode() != ValueOpcode::GetBufferResource ||
				       std::ranges::any_of(use.user->Uses(), [&](const Use& consumer) {
					       return !IndirectBufferAccess(*consumer.user);
				       });
			    })) {
				plan.retain_reads = true;
			}
			plan.memory[dword] = memory_index;
			plan.reads[dword] = read;
		}

		return true;
	}

	bool SetExactTableMaximum(Value key, DescriptorSource::IndirectDescriptor& indirect,
	                          const Inst* handle = nullptr) {
		const auto* lane = key.Resolve().TryInstruction();
		if (lane != nullptr && lane->GetOpcode() != ValueOpcode::ReadFirstLane) lane = nullptr;
		SelectorDomain result {0u};
		if (handle != nullptr) {
			if (handle->Uses().empty()) return false;
			for (const auto& use: handle->Uses()) {
				if (ImageOpcodeInfoOf(use.user->GetOpcode()).access == ImageAccess::None) return false;
				const auto active = use.user->Arg(use.user->NumArgs() - 1u);
				if (lane != nullptr) {
					if (!Implies(active, lane->Arg(1))) return false;
				} else result.maximum = std::max(result.maximum, SelectorMaximum(key, {}, 0u, active));
			}
		}
		if (lane != nullptr) {
			result = SelectedSelectorDomain(*lane, handle != nullptr ? lane->Arg(1) : Value {});
			if (result.maximum == UINT32_MAX &&
			    (handle != nullptr || HasActiveLane(lane->Arg(1), lane->Parent())))
				result.maximum = SelectorMaximum(lane->Arg(0), {}, 0u, lane->Arg(1));
		} else if (handle == nullptr) {
			result.maximum = SelectorMaximum(key);
		}
		if (result.maximum > UINT32_MAX / 4u ||
		    uint64_t {result.maximum} * indirect.table_stride + indirect.table_offset > UINT32_MAX)
			return false;
		indirect.table_maximum = result.maximum;
		indirect.key_count = result.count;
		return true;
	}

	bool TryMakeNestedDescriptor(Inst& handle, const DescriptorSource& descriptor,
	                             IndirectDescriptorPlan& plan) {
		if (handle.GetOpcode() != ValueOpcode::GetImageResource &&
		    handle.GetOpcode() != ValueOpcode::GetSamplerResource) return false;
		auto reads = plan;
		Value child_key;
		uint32_t offset = 0, stride = 0, immediate = 0;
		if (!MatchDescriptorReads(handle, descriptor, reads, child_key, offset, stride, immediate, true) ||
		    m_program.memory_info[reads.memory[0]].kind != ResourceKind::ScalarAddress ||
		    uint64_t {offset & ~3u} + (immediate & ~3u) > UINT32_MAX) return false;
		const auto* pointer = reads.reads[0]->Arg(0).ResolveInstruction();
		DescriptorSource::IndirectDescriptor indirect;
		DescriptorSource::IndirectDescriptor::PointerTable child {
		    .offset = (offset & ~3u) + (immediate & ~3u), .stride = stride};
		DescriptorSource table_source;
		Value key;
		if (child_key.IsEmpty()) {
			// Missing scalar-buffer pointer words are zero-filled. Only the existing
			// unmapped first page permits omitted rows to map directly to a null image.
			if (uint64_t {child.offset} + plan.read_dwords * 4u > 0x4000u) return false;
			const auto* low = pointer->Arg(0).Resolve().TryInstruction();
			const auto* high = pointer->Arg(1).Resolve().TryInstruction();
			uint32_t low_index, high_index;
			const auto* low_memory = low != nullptr ? ScalarReadMemory(*low, low_index) : nullptr;
			const auto* high_memory = high != nullptr ? ScalarReadMemory(*high, high_index) : nullptr;
			if (low_memory == nullptr || high_memory == nullptr ||
			    low_memory->kind != ResourceKind::ScalarBuffer ||
			    high_memory->kind != ResourceKind::ScalarBuffer ||
			    low_memory->offset > INT32_MAX || high_memory->offset > INT32_MAX ||
			    (low_memory->offset & ~3u) + 4u != (high_memory->offset & ~3u) ||
			    !EquivalentValue(m_program, low->Arg(0), high->Arg(0)) ||
			    !EquivalentValue(m_program, low->Arg(1), high->Arg(1)) ||
			    !AvailableBefore(*low, *pointer) || !AvailableBefore(*high, *pointer) ||
			    !MatchTableOffset(low->Arg(1), key, indirect.table_offset, indirect.table_stride) ||
			    indirect.table_stride < 8u || (indirect.table_stride & 3u) != 0u ||
			    !SetExactTableMaximum(key, indirect) ||
			    (key.TryInstruction() != nullptr && !AvailableBefore(*key.Instruction(), handle)) ||
			    !MakeRuntimeTableSource(*low, table_source)) return false;
			indirect.table_immediate = low_memory->offset & ~3u;
		} else {
			if (handle.GetOpcode() != ValueOpcode::GetImageResource ||
			    !ImmediateU32(UnsignedResourceBound(child_key, handle), child.count) || child.count == 0u ||
			    child.count > MaxIndirectDescriptorProbes || stride < plan.read_dwords * 4u ||
			    ((offset | stride) & 3u) != 0u ||
			    uint64_t {child.offset} + uint64_t {child.count - 1u} * stride +
			        plan.read_dwords * 4u > UINT32_MAX) return false;
			const auto* low = pointer->Arg(0).Resolve().TryInstruction();
			const auto* high = pointer->Arg(1).Resolve().TryInstruction();
			if (low == nullptr || high == nullptr || low->GetOpcode() != ValueOpcode::ReadFirstLane ||
			    high->GetOpcode() != ValueOpcode::ReadFirstLane ||
			    !EquivalentValue(m_program, low->Arg(1), high->Arg(1))) return false;
			const auto active = low->Arg(1);
			const auto* low_word = ActiveValue(low->Arg(0), active).TryInstruction();
			auto high_value = ActiveValue(high->Arg(0), active);
			const auto* extended = high_value.TryInstruction();
			// Scalar addresses use the low 48 bits, so signed high-WORD extension
			// preserves exactly the same pointer as the original pair in memory.
			if (extended != nullptr && extended->GetOpcode() == ValueOpcode::BitFieldSExtract &&
			    extended->Arg(1).Resolve() == Value(0u) && extended->Arg(2).Resolve() == Value(16u))
				high_value = ActiveValue(extended->Arg(0), active);
			const auto* high_word = high_value.TryInstruction();
			if (low_word == nullptr || high_word == nullptr ||
			    low_word->GetOpcode() != ValueOpcode::CompositeExtractU32x2 ||
			    high_word->GetOpcode() != ValueOpcode::CompositeExtractU32x2 ||
			    low_word->Arg(1).Resolve() != Value(0u) || high_word->Arg(1).Resolve() != Value(1u) ||
			    low_word->Arg(0).Resolve() != high_word->Arg(0).Resolve()) return false;
			const auto* read = low_word->Arg(0).Resolve().TryInstruction();
			if (read == nullptr || read->GetOpcode() != ValueOpcode::LoadBufferU32x2 ||
			    !Implies(active, read->Arg(4))) return false;
			const auto& memory = m_program.memory_info[read->Flags<MemoryFlags>().index];
			if (memory.kind != ResourceKind::Buffer || memory.typed || memory.formatted ||
			    !memory.idxen || memory.offen || memory.data_bits != 32u || memory.data_dwords != 2u ||
			    (memory.offset & 3u) != 0u || memory.offset > UINT32_MAX - 8u ||
			    read->Arg(2).Resolve() != Value(0u) || read->Arg(3).Resolve() != Value(0u)) return false;
			for (const auto& use: handle.Uses()) {
				if (use.operand != 0u || ImageOpcodeInfoOf(use.user->GetOpcode()).access == ImageAccess::None)
					return false;
				const auto executing = use.user->Arg(use.user->NumArgs() - 1u);
				if (!Implies(executing, active)) return false;
				const auto outer = EqualLocalKey(executing, read->Arg(1));
				const auto* lane = outer.TryInstruction();
				if (lane == nullptr || lane->GetOpcode() != ValueOpcode::ReadFirstLane ||
				    !EquivalentValue(m_program, lane->Arg(0), read->Arg(1)) ||
				    !EquivalentValue(m_program, lane->Arg(1), active) ||
				    !AvailableBefore(*lane, handle) ||
				    (!key.IsEmpty() && !EquivalentValue(m_program, key, outer))) return false;
				key = outer;
			}
			if (!SetExactTableMaximum(key, indirect, &handle) ||
			    (uint64_t {*indirect.table_maximum} + 1u) * child.count > MaxIndirectDescriptorProbes ||
			    !MakeRuntimeTableSource(*read, table_source)) return false;
			indirect.table_immediate = memory.offset;
			child.indexed = true;
		}
		if (table_source.dword_count != 4u) return false;
		indirect.table_source = InternSource(table_source);
		indirect.table_scalar = true;
		indirect.pointer_table = child;
		DescriptorSource source;
		source.dword_count = handle.NumArgs();
		source.dwords.fill(Value(0u));
		std::copy_n(table_source.dwords.begin(), 4u, source.dwords.begin());
		source.indirect_descriptor = indirect;
		if (!child_key.IsEmpty()) {
			auto* block = handle.Parent();
			const auto where = std::ranges::find_if(block->Instructions(),
			    [&](const Inst& inst) { return &inst == &handle; });
			key = Value(&*block->PrependNewInst(where, ValueOpcode::IMul32, {key, Value(child.count)}));
			key = Value(&*block->PrependNewInst(where, ValueOpcode::IAdd32, {key, child_key}));
		}
		plan.handle = &handle;
		plan.source = InternSource(source);
		plan.key = key;
		plan.roots = source.dwords;
		return true;
	}

	// The EXEC predicate of the image access through use; empty for any other use.
	Value AccessPredicate(const Use& use, bool sampler) const {
		const auto& access = *use.user;
		if (use.operand != (sampler ? 1u : 0u) ||
		    ImageOpcodeInfoOf(access.GetOpcode()).access == ImageAccess::None) return {};
		if (sampler && (access.GetOpcode() != ValueOpcode::ImageSampleRaw ||
		    (m_program.memory_info[access.Flags<MemoryFlags>().index].image_sample_flags &
		     Decoder::ImageSampleFlagAdjust) != 0u)) return {};
		const auto mask = access.Arg(access.NumArgs() - 1u);
		return mask.GetType() == Type::U1 ? mask : Value {};
	}

	// A key no bound covers takes its maximum from a signed clamp to a constant, on the key or on
	// the key plus a constant in an earlier-tracked sibling read of the same table. Negative keys
	// and keys past that maximum read the first row.
	uint32_t ClampedTableMaximum(Value key, const DescriptorSource& table,
	                             const DescriptorSource::IndirectDescriptor& indirect) const {
		uint32_t row = 0, offset = 0;
		const auto clamped = [&](Value value) {
			const auto* min = value.Resolve().TryInstruction();
			if (min != nullptr && min->GetOpcode() == ValueOpcode::SMin32)
				for (uint32_t arg = 0; arg < 2u; ++arg)
					if (ImmediateU32(min->Arg(arg), row)) return min->Arg(arg ^ 1u);
			return Value {};
		};
		if (!clamped(key).IsEmpty()) return row;
		for (const auto& sibling: m_indirect_descriptors) {
			const auto& read = *m_sources[sibling.source].indirect_descriptor;
			const auto* add = clamped(sibling.key).Resolve().TryInstruction();
			if (add == nullptr || add->GetOpcode() != ValueOpcode::IAdd32 ||
			    m_sources[read.table_source] != table ||
			    read.table_offset != indirect.table_offset ||
			    read.table_stride != indirect.table_stride ||
			    read.table_immediate != indirect.table_immediate) continue;
			for (uint32_t arg = 0; arg < 2u; ++arg)
				if (ImmediateU32(add->Arg(arg), offset) &&
				    EquivalentValue(m_program, add->Arg(arg ^ 1u), key)) return row;
		}
		return UINT32_MAX;
	}

	bool TryMakeTableDescriptor(Inst& handle, const DescriptorSource& descriptor,
	                            IndirectDescriptorPlan& plan) {
		const bool sampler = handle.GetOpcode() == ValueOpcode::GetSamplerResource;
		if ((!sampler && handle.GetOpcode() != ValueOpcode::GetImageResource) ||
		    handle.NumArgs() != (sampler ? 4u : 8u)) return false;
		DescriptorSource table_source;
		Value key;
		uint32_t table_offset = 0;
		uint32_t table_stride = 0;
		uint32_t table_immediate = 0;
		if (!MatchDescriptorReads(handle, descriptor, plan, key, table_offset,
		                          table_stride, table_immediate) ||
		    !MakeRuntimeTableSource(*plan.reads[0], table_source)) return false;
		DescriptorSource material_source;
		DescriptorSource::IndirectDescriptor indirect;
		indirect.table_offset = table_offset;
		indirect.table_immediate = table_immediate;
		indirect.table_stride = table_stride;
		indirect.workgroup_axis = WorkgroupAxis(key);
		if (indirect.workgroup_axis != UINT32_MAX && !plan.table_indexed) {
			// The descriptor also supplies dimensions to shader arithmetic. Keep its reads;
			// only the image handle is projected onto the bounded workgroup key.
			plan.retain_reads = true;
		} else if (!plan.table_indexed &&
		           MatchScalarMaterialKey(key, indirect, material_source,
		                                  !sampler && table_source.dword_count == 4u, sampler)) {
			indirect.selector->excluded = ExcludedKeyOnEntry(key, handle.Parent());
			// Keep live descriptor payloads, but project descriptor-only SMEM onto
			// the GPU key using the same plan as other indirect image reads.
			plan.retain_reads |= table_source.dword_count == 4u;
		} else if (!sampler && !plan.table_indexed && table_source.dword_count == 4u &&
		           MatchUniformizedMaterialKey(key, handle, indirect, material_source, MaterialKey::Heap)) {
			// Preserve the heap's byte-offset key so runtime geometry can choose the
			// tighter selector domain without changing the GPU's descriptor reads.
			indirect.table_scalar = true;
			plan.table_read = plan.reads[0];
			plan.retain_reads = true;
		} else if (sampler && table_source.dword_count == 4u && !plan.table_indexed &&
		           SetExactTableMaximum(key, indirect, &handle)) {
			indirect.table_scalar = true;
			plan.retain_reads = true;
		} else if (table_source.dword_count == 2u) {
			auto guarded_count = UnsignedBoundOnEntry(key, handle.Parent());
			if (guarded_count.IsEmpty()) guarded_count = UnsignedResourceBound(key, handle);
			// A runtime count can describe fewer rows than the mask's static bit range.
			if (guarded_count.IsEmpty()) guarded_count = UnsignedBoundOnEntry(key, handle.Parent(), true);
			if (!guarded_count.IsEmpty()) {
				if (table_stride < plan.read_dwords * 4u || (table_stride & 3u) != 0u) return false;
				// Materialization captures the count and descriptor reads; renderer alias
				// checks reject writes to those ranges while unrelated stores remain valid.
				indirect.key_count = guarded_count;
			} else if (table_stride != plan.read_dwords * 4u) {
				return false;
			} else {
				uint32_t first = 0;
				const auto* bound = BoundedInduction(key, handle.Parent(),
				    [&](uint32_t initial, uint32_t step) {
					    first = initial;
					    return initial <= INT32_MAX && step == 1u;
				    }, [&](const Inst& compare, uint32_t) {
					    return (compare.GetOpcode() == ValueOpcode::SLessThan32 ||
					            (first == 0u && compare.GetOpcode() == ValueOpcode::ULessThan32)) &&
					           ValidateRuntimeValue(m_program, compare.Arg(1), RuntimeValueType::Integer);
				    });
				if (bound != nullptr) {
					indirect.key_count = bound->Arg(1);
					indirect.signed_count = bound->GetOpcode() == ValueOpcode::SLessThan32;
					if (first != 0u) {
						// Clamp before enumerating: negative signed bounds have no reachable keys.
						auto& nonempty = m_program.value_storage.emplace_back(ValueOpcode::SGreaterThanEqual32);
						nonempty.SetArg(0, bound->Arg(1));
						nonempty.SetArg(1, Value(first));
						auto& length = m_program.value_storage.emplace_back(ValueOpcode::ISub32);
						length.SetArg(0, bound->Arg(1));
						length.SetArg(1, Value(first));
						auto& count = m_program.value_storage.emplace_back(ValueOpcode::SelectU32);
						count.SetArg(0, Value(&nonempty));
						count.SetArg(1, Value(&length));
						count.SetArg(2, Value(0u));
						indirect.key_count = Value(&count);
						indirect.selector_first = Value(first);
						indirect.signed_count = false;
					}
				}
			}
			if (indirect.key_count.IsEmpty() &&
			    !MatchUniformizedMaterialKey(key, handle, indirect, material_source, MaterialKey::Global) &&
			    !MatchUniformizedMaterialKey(key, handle, indirect, material_source, MaterialKey::Indexed) &&
			    !MatchReversedImageRange(key, indirect) && !MatchWorkgroupKey(key, indirect)) {
				auto maximum = SelectorMaximum(key);
				if (maximum >= MaxIndirectDescriptorProbes) {
					// Each image access bounds the key on the lanes it executes.
					maximum = 0u;
					for (const auto& use: handle.Uses()) {
						const auto active = AccessPredicate(use, false);
						if (active.IsEmpty()) {
							maximum = UINT32_MAX;
							break;
						}
						maximum = std::max(maximum, SelectorMaximum(key, {}, 0, active));
					}
				}
				if (maximum >= MaxIndirectDescriptorProbes)
					maximum = ClampedTableMaximum(key, table_source, indirect);
				// A static key range can cover other operations' slots, so samples keep their GPU key
				// over an image heap, which reads slots of another type or invalid T#s as null.
				if (maximum < MaxIndirectDescriptorProbes && !sampler &&
				    std::ranges::all_of(handle.Uses(), [](const Use& use) {
					    return use.user->GetOpcode() == ValueOpcode::ImageSampleRaw;
				    })) {
					indirect.table_maximum = maximum;
					indirect.table_scalar = true;
					plan.table_read = plan.reads[0];
					plan.retain_reads = true;
				} else if (maximum < MaxIndirectDescriptorProbes)
					indirect.key_count = Value(maximum + 1u);
				else if (!ProgramLoopCount(key, indirect) && (sampler ||
				         !MatchUniformizedMaterialKey(key, handle, indirect, material_source, MaterialKey::Guarded)))
					return false;
			}
			if ((table_offset & 3u) != 0u) return false;
		} else {
			// Keep the GPU selector and descriptor reads, and preserve the proven scalar
			// offset domain within the bounded image table.
			if (sampler || plan.split_offsets) return false;
			bool shared_selector = false;
			if (!plan.table_indexed) {
				for (const auto& owner: m_indirect_descriptors) {
					const auto& selected = *m_sources[owner.source].indirect_descriptor;
					if (!selected.selector || !EquivalentValue(m_program, key, owner.key) ||
					    !std::ranges::all_of(handle.Uses(), [&](const Use& target) {
						    const auto target_active = AccessPredicate(target, false);
						    return !target_active.IsEmpty() &&
						        std::ranges::any_of(owner.handle->Uses(), [&](const Use& witness) {
							        const auto owner_active = AccessPredicate(witness,
							            owner.handle->GetOpcode() == ValueOpcode::GetSamplerResource);
							        return !owner_active.IsEmpty() && AvailableBefore(*witness.user, *target.user) &&
							            Implies(target_active, owner_active);
						        });
					    })) continue;
					// These exact keys are already required wherever this image is accessed.
					// Reuse their proven domain without adding selector reads or a second plan.
					indirect.selector = selected.selector;
					indirect.key_count = selected.key_count;
					indirect.selector_first = selected.selector_first;
					indirect.selector_mask = selected.selector_mask;
					indirect.signed_count = selected.signed_count;
					indirect.workgroup_axis = selected.workgroup_axis;
					if (selected.table_scalar) {
						indirect.table_scalar = true;
						plan.table_read = plan.reads[0];
					}
					material_source = m_sources[selected.selector->source];
					shared_selector = true;
					break;
				}
			}
			if (!shared_selector) {
				indirect.workgroup_axis = UINT32_MAX;
				plan.table_read = plan.reads[0];
				if (plan.table_indexed) {
					indirect.table_stride = 0u;
					indirect.table_offset = 0u;
					indirect.table_immediate = 0u;
					indirect.table_record_bytes = table_immediate + plan.read_dwords * 4u;
				} else {
					indirect.table_scalar = true;
				}
			}
			plan.retain_reads = true;
		}

		if (plan.retain_reads) plan.reads.fill(nullptr);
		indirect.table_source = InternSource(table_source);
		DescriptorSource source;
		source.dword_count = handle.NumArgs();
		source.dwords.fill(Value(0u));
		std::copy_n(material_source.dwords.begin(), material_source.dword_count,
		            source.dwords.begin());
		if (!sampler) std::copy_n(table_source.dwords.begin(), table_source.dword_count,
		                          source.dwords.begin() + 4u);
		source.indirect_descriptor = indirect;
		plan.handle = &handle;
		plan.source = InternSource(source);
		plan.key = key;
		plan.roots = source.dwords;
		return true;
	}

	bool MatchPackedScalarKey(Value key, DescriptorSource::IndirectDescriptor& indirect) {
		const auto* mask = key.Resolve().TryInstruction();
		uint32_t bits = 0;
		if (mask == nullptr || mask->GetOpcode() != ValueOpcode::BitwiseAnd32 ||
		    !ImmediateU32(mask->Arg(1), bits) || bits != 255u) return false;
		const auto* shift = mask->Arg(0).Resolve().TryInstruction();
		if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftRightLogical32 ||
		    (PossibleU32Bits(shift->Arg(1)) & ~24u) != 0u) return false;
		DescriptorSource::IndirectDescriptor::SelectorRead selector;
		for (const auto node: PhiWeb(shift->Arg(0))) {
			const auto* read = node.TryInstruction();
			if (read == nullptr) return false;
			if (read->GetOpcode() == ValueOpcode::Phi) continue;
			uint32_t index = 0;
			const auto* memory = ScalarReadMemory(*read, index);
			DescriptorSource source;
			if (memory == nullptr || memory->kind != ResourceKind::ScalarAddress ||
			    memory->offset > INT32_MAX || !MakeRuntimeTableSource(*read, source)) return false;
			const auto maximum = PossibleU32Bits(read->Arg(1)) & ~3u;
			const auto offset = memory->offset & ~3u;
			if (uint64_t {offset} + maximum + 4u > UINT32_MAX) return false;
			const auto source_index = InternSource(source);
			if (selector.source != UINT32_MAX && selector.source != source_index) return false;
			selector.source = source_index;
			selector.byte_ranges.emplace_back(offset, maximum + 4u);
		}
		if (selector.byte_ranges.empty()) return false;
		// Scalar reads align both offsets independently. Merge the possible read spans;
		// the GPU retains the actual byte selection and the host only discovers V#s.
		std::ranges::sort(selector.byte_ranges);
		for (size_t i = 1; i < selector.byte_ranges.size();) {
			auto& previous = selector.byte_ranges[i - 1u];
			const auto current = selector.byte_ranges[i];
			if (current.first > previous.first + previous.second) { ++i; continue; }
			previous.second = std::max(previous.second, current.first + current.second - previous.first);
			selector.byte_ranges.erase(selector.byte_ranges.begin() + i);
		}
		indirect.selector = std::move(selector);
		return true;
	}

	// A lane may retain a previously selected row across loop iterations. Each assignment
	// must supply either zero or a value explicitly checked against the same uniform count.
	Value GuardedLaneCount(Value key, uint32_t maximum) {
		const auto* lane = key.Resolve().TryInstruction();
		if (lane == nullptr || lane->GetOpcode() != ValueOpcode::ReadFirstLane) return {};
		auto selected = lane->Arg(0).Resolve();
		uint32_t limit = maximum == UINT32_MAX ? UINT32_MAX : maximum + 1u;
		while (const auto* choice = selected.TryInstruction()) {
			if (choice->GetOpcode() != ValueOpcode::SelectU32 || choice->Arg(2).Resolve() != Value(0u))
				break;
			const auto bound = UnsignedBound(choice->Arg(0), choice->Arg(1));
			uint32_t immediate;
			if (!bound.IsEmpty()) {
				if (!ImmediateU32(bound, immediate)) break;
				limit = std::min(limit, immediate);
			}
			selected = choice->Arg(1).Resolve();
		}
		Value count;
		std::vector<Value> pending {selected};
		for (size_t i = 0; i < pending.size(); ++i) {
			if (pending[i] == Value(0u)) continue;
			const auto* choice = pending[i].TryInstruction();
			if (choice == nullptr) return {};
			const bool select = choice->GetOpcode() == ValueOpcode::SelectU32;
			if (!select && (choice->GetOpcode() != ValueOpcode::Phi || choice->NumArgs() == 0u)) return {};
			for (size_t arm = select ? 1u : 0u; arm < choice->NumArgs(); ++arm) {
				const auto value = choice->Arg(arm).Resolve();
				const auto bound = select ? UnsignedBound(choice->Arg(0), value, arm == 1u) : Value {};
				if (!bound.IsEmpty() && value != Value(0u)) {
					if (!ValidateRuntimeValue(m_program, bound, RuntimeValueType::Integer) ||
					    (!count.IsEmpty() && !EquivalentValue(m_program, count, bound))) return {};
					count = bound;
				} else if (std::ranges::find(pending, value) == pending.end()) {
					pending.push_back(value);
				}
			}
		}
		if (count.IsEmpty()) return {};
		const auto adjust = [&](ValueOpcode op, uint32_t immediate) {
			auto& node = m_program.value_storage.emplace_back(op);
			node.SetArg(0, count);
			node.SetArg(1, Value(immediate));
			count = Value(&node);
		};
		if (limit != UINT32_MAX) adjust(ValueOpcode::UMin32, limit);
		adjust(ValueOpcode::UMax32, 1u);
		return count;
	}

	bool TryMakeIndirectBuffer(Inst& handle, const DescriptorSource& descriptor,
	                           IndirectDescriptorPlan& plan) {
		if (handle.GetOpcode() != ValueOpcode::GetBufferResource || handle.NumArgs() != 4u ||
		    handle.Uses().empty()) return false;
		if (!IndirectBufferAccess(*handle.Uses().front().user)) return false;
		const bool legacy_store = handle.Uses().front().user->GetOpcode() == ValueOpcode::StoreBufferU32;
		const auto indexed = m_program.memory_info[handle.Uses().front().user->Flags<MemoryFlags>().index].idxen;
		for (const auto& use: handle.Uses()) {
			const auto& inst = *use.user;
			if (!IndirectBufferAccess(inst)) return false;
			const auto& memory = m_program.memory_info[inst.Flags<MemoryFlags>().index];
			if (memory.idxen != indexed) return false;
		}
		DescriptorSource table_source;
		Value key;
		uint32_t table_offset = 0;
		uint32_t table_stride = 0;
		uint32_t table_immediate = 0;
		if (!MatchDescriptorReads(handle, descriptor, plan, key, table_offset,
		                          table_stride, table_immediate) ||
		    !MakeRuntimeTableSource(*plan.reads[0], table_source) ||
		    table_stride < 16u || (table_stride & 3u) != 0u) return false;
		DescriptorSource::IndirectDescriptor indirect;
		indirect.table_stride = table_stride;
		indirect.table_offset = table_offset;
		indirect.table_immediate = table_immediate;
		indirect.indexed = indexed;
		for (uint32_t word = 0; word < 4u; ++word) {
			const auto* anchor = plan.reads[word];
			const auto* key_inst = key.Resolve().TryInstruction();
			if (AvailableBefore(*anchor, handle) &&
			    (key_inst == nullptr || (key_inst != anchor && AvailableBefore(*key_inst, *anchor))) &&
			    std::all_of(plan.reads.begin(), plan.reads.begin() + 4u,
			                [&](const Inst* read) { return AvailableBefore(*read, *anchor); })) {
				plan.selection_anchor = anchor;
				break;
			}
		}
		if (!legacy_store && plan.selection_anchor == nullptr) return false;
		const auto finish = [&] {
			indirect.table_source = InternSource(table_source);
			DescriptorSource source;
			source.dword_count = 4u;
			source.dwords.fill(Value(0u));
			source.indirect_descriptor = std::move(indirect);
			plan.handle = &handle;
			plan.source = InternSource(source);
			plan.key = key;
			return true;
		};
		if (const auto* bound = BoundedLoop(key, handle.Parent()); bound != nullptr) {
			indirect.key_count = bound->Arg(1);
			indirect.signed_count = bound->GetOpcode() == ValueOpcode::SLessThan32;
			return finish();
		}
		if (table_source.dword_count == 4u && !plan.split_offsets &&
		    plan.selection_anchor != nullptr && SetExactTableMaximum(key, indirect)) {
			indirect.table_scalar = true;
			return finish();
		}
		DescriptorSource material_source;
		if (!plan.split_offsets && MatchScalarMaterialKey(key, indirect, material_source, true)) {
			plan.retain_reads = true;
			return finish();
		}
		const auto axis = table_source.dword_count == 2u ? WorkgroupAxis(key) : UINT32_MAX;
		if (!plan.split_offsets && (!legacy_store || axis != UINT32_MAX)) {
			indirect.workgroup_axis = axis;
			indirect.key_count = UnsignedResourceBound(key, handle);
			if (axis != UINT32_MAX || !indirect.key_count.IsEmpty()) return finish();
		}
		if (!legacy_store) {
			if (table_source.dword_count == 2u && !plan.split_offsets) {
				const auto maximum = SelectorMaximum(key);
				indirect.key_count = GuardedLaneCount(key, maximum);
				if (!indirect.key_count.IsEmpty()) return finish();
				if (maximum >= MaxIndirectDescriptorProbes) return false;
				indirect.key_count = Value(maximum + 1u);
				return finish();
			}
			// A bounded V# supplies every reachable descriptor without reading the GPU key.
			// Separate U32 offsets could wrap each DWORD to a different table position.
			if (table_source.dword_count != 4u || plan.split_offsets) return false;
			indirect.table_scalar = true;
			return finish();
		}
		if (table_stride != 16u) return false;
		if (table_source.dword_count == 2u) {
			if (!MatchPackedScalarKey(key, indirect)) {
				// EXEC implications must describe the same iteration as the selected value.
				if (m_native_cfg.irreducible || !m_native_cfg.back_edges.empty()) return false;
				const auto* lane = key.Resolve().TryInstruction();
				if (lane == nullptr || lane->GetOpcode() != ValueOpcode::ReadFirstLane ||
				    !std::ranges::all_of(handle.Uses(), [&](const Use& use) {
					    return Implies(use.user->Arg(use.user->NumArgs() - 1u), lane->Arg(1));
				    })) return false;
				const auto maximum = SelectorMaximum(lane->Arg(0), {}, 0u, lane->Arg(1));
				if (maximum >= MaxIndirectDescriptorProbes) return false;
				indirect.key_count = Value(maximum + 1u);
			}
			return finish();
		}
		if (table_source.dword_count != 4u || table_offset != 0u || table_immediate != 0u || indexed)
			return false;
		const auto* lane_read = key.Resolve().TryInstruction();
		if (lane_read == nullptr || lane_read->GetOpcode() != ValueOpcode::ReadFirstLane)
			return false;
		const Inst* selected = nullptr;
		for (const auto& use: handle.Uses()) {
			const auto* value = UniformizedMaterialValue(
			    key, *use.user, use.user->Arg(use.user->NumArgs() - 1u));
			if (value == nullptr || (selected != nullptr && selected != value)) return false;
			selected = value;
		}
		DescriptorSource::IndirectDescriptor::SelectorRead selector;
		const auto* read = IndexedMaterialRead(selected->Arg(1).Resolve().TryInstruction(),
		                                       selected->Arg(0), selector);
		if (read == nullptr || !MakeRuntimeTableSource(*read, material_source) ||
		    !BoundedSelectedIndex(read->Arg(1), read->Arg(4), indirect)) return false;
		selector.source = InternSource(material_source);
		indirect.selector = std::move(selector);
		return finish();
	}


	bool SelectorPredicateFalse(Value value, SelectorBounds bounds, uint32_t depth = 0,
	                            Value guard = {}) const {
		if (depth > 32u) return false;
		value = value.Resolve();
		if (value.IsImmediate()) return value.GetType() == Type::U1 && !value.U1();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd)
			return SelectorPredicateFalse(inst->Arg(0), bounds, depth + 1u, guard) ||
			       SelectorPredicateFalse(inst->Arg(1), bounds, depth + 1u, guard);
		if (inst->GetOpcode() == ValueOpcode::IEqual32) {
			for (uint32_t arg = 0; arg < 2u; ++arg) {
				uint32_t target;
				if (ImmediateU32(inst->Arg(arg), target) &&
				    SelectorMaximum(inst->Arg(arg ^ 1u), bounds, depth + 1u, guard) < target) return true;
			}
		}
		uint32_t target;
		return inst->GetOpcode() == ValueOpcode::ULessThan32 && ImmediateU32(inst->Arg(0), target) &&
		       SelectorMaximum(inst->Arg(1), bounds, depth + 1u, guard) <= target;
	}

	struct SelectorDomain {
		uint32_t maximum = UINT32_MAX;
		Value count;
	};

	SelectorDomain FiniteSelectorDomain(Value value, Value guard) const {
		value = value.Resolve();
		bool nonnegative = false;
		if (const auto* root = value.TryInstruction()) {
			for (const auto& use: root->Uses()) {
				const auto& compare = *use.user;
				const auto op = compare.GetOpcode();
				if (((op == ValueOpcode::SLessThanEqual32 && use.operand == 1u &&
				      compare.Arg(0).Resolve() == Value(0u)) ||
				     (op == ValueOpcode::SGreaterThanEqual32 && use.operand == 0u &&
				      compare.Arg(1).Resolve() == Value(0u))) &&
				    Implies(guard, Value(use.user))) nonnegative = true;
			}
		}
		struct Arm { Value value; const Inst* choice = nullptr; bool positive = true; };
		std::vector<Arm> pending {{value}};
		std::vector<const Inst*> visited;
		uint32_t maximum = 0u;
		bool found = false;
		bool counted = true;
		Value count;
		for (size_t at = 0; at < pending.size(); ++at) {
			uint32_t literal;
			const auto current = ActiveValue(pending[at].value, guard);
			if (ImmediateU32(current, literal)) {
				if (nonnegative && std::bit_cast<int32_t>(literal) < 0) continue;
				maximum = std::max(maximum, literal);
				found = true;
				if (counted) {
					Value candidate;
					if (const auto* choice = pending[at].choice; choice != nullptr) {
						if (choice->GetOpcode() == ValueOpcode::SelectU32)
							candidate = UnsignedBound(choice->Arg(0), current, pending[at].positive,
							    false, pending[at].positive ? choice->Arg(0) : Value {});
						if (candidate.IsEmpty() && choice->Parent() != nullptr)
							candidate = UnsignedBoundOnEntry(current, choice->Parent());
					}
					if (candidate.IsEmpty() || !ValidateRuntimeValue(m_program, candidate, RuntimeValueType::Integer) ||
					    (!count.IsEmpty() && !EquivalentValue(m_program, count, candidate))) counted = false;
					else count = candidate;
				}
				continue;
			}
			const auto* choice = current.TryInstruction();
			if (choice == nullptr) return {};
			const bool phi = choice->GetOpcode() == ValueOpcode::Phi;
			if ((!phi && choice->GetOpcode() != ValueOpcode::SelectU32) ||
			    (phi && (choice->NumArgs() == 0u || choice->Parent() == nullptr ||
			             InCycle(choice->Parent())))) return {};
			// Each literal uses only its own choice and block-entry proof. A shared
			// choice therefore cannot borrow a bound from a different parent path.
			if (std::ranges::find(visited, choice) != visited.end()) continue;
			visited.push_back(choice);
			for (size_t arm = phi ? 0u : 1u; arm < choice->NumArgs(); ++arm)
				pending.push_back({choice->Arg(arm), choice, arm == 1u});
		}
		return found ? SelectorDomain {maximum, counted ? count : Value {}} : SelectorDomain {};
	}

	SelectorDomain SelectedSelectorDomain(const Inst& lane, Value guard) const {
		if (guard.IsEmpty() ? !HasActiveLane(lane.Arg(1), lane.Parent())
		                    : !Implies(guard, lane.Arg(1))) return {};
		auto result = FiniteSelectorDomain(lane.Arg(0), lane.Arg(1));
		if (result.maximum == UINT32_MAX) result.maximum = PossibleU32Bits(lane.Arg(0));
		return result;
	}

	// Bounds of an F32 value on the lanes where guard holds; NaN when unknown.
	std::pair<float, float> FloatRange(Value value, Value guard, uint32_t depth = 0) const {
		constexpr std::pair unknown {NAN, NAN};
		value = ActiveValue(value, guard);
		if (value.IsImmediate() && value.GetType() == Type::F32)
			return {value.F32Value(), value.F32Value()};
		const auto* inst = value.TryInstruction();
		// Each multiply-add evaluates three operands; keep the search shallow.
		if (depth > 12u || inst == nullptr) return unknown;
		const auto range = [&](uint32_t arg) {
			return FloatRange(inst->Arg(arg), guard, depth + 1u);
		};
		switch (inst->GetOpcode()) {
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::BitCastU32F32: return range(0);
			// Clamping maps NaN to zero.
			case ValueOpcode::FPSaturate32: return {0.0f, 1.0f};
			case ValueOpcode::FPFloor32: {
				const auto [low, high] = range(0);
				return {std::floor(low), std::floor(high)};
			}
			case ValueOpcode::FPMad32: {
				// The rounded multiply and add are monotonic, so the bounds lie at the corners.
				const auto a = range(0), b = range(1), c = range(2);
				if (!std::isfinite(a.first) || !std::isfinite(a.second) ||
				    !std::isfinite(b.first) || !std::isfinite(b.second)) return unknown;
				const auto [low, high] = std::minmax({a.first * b.first, a.first * b.second,
				                                      a.second * b.first, a.second * b.second});
				const std::pair bounds {low + c.first, high + c.second};
				if (!std::isfinite(bounds.first) || !std::isfinite(bounds.second)) return unknown;
				return bounds;
			}
			default: return unknown;
		}
	}

	uint32_t SelectorMaximum(Value value, SelectorBounds bounds = {}, uint32_t depth = 0,
	                         Value guard = {}) const {
		if (depth > 32u) return UINT32_MAX;
		value = value.Resolve();
		if (value.IsImmediate() && value.GetType() == Type::U32) return value.U32();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return UINT32_MAX;
		for (const auto& [phi, maximum]: bounds)
			if (inst == phi) return maximum;
		const auto maximum = [&](uint32_t arg) {
			return SelectorMaximum(inst->Arg(arg), bounds, depth + 1u, guard);
		};
		switch (inst->GetOpcode()) {
			case ValueOpcode::ReadFirstLane:
				return SelectedSelectorDomain(*inst, guard).maximum;
			case ValueOpcode::ReadLane: {
				// ReadLane ignores EXEC; only context-free bounds on every source lane apply,
				// unless guard proves a waterfall key equal to the executing lanes' local value.
				const auto local = EqualLocalKey(guard, value);
				return local.IsEmpty() ? PossibleU32Bits(inst->Arg(0))
				                       : std::min(SelectorMaximum(local, bounds, depth + 1u, guard),
				                                  PossibleU32Bits(inst->Arg(0)));
			}
			case ValueOpcode::Phi: {
				const auto initial = InitialCandidateMask(value);
				if (!initial.IsEmpty()) return SelectorMaximum(initial, bounds, depth + 1u, guard);
				const auto post_test = PostTestLoopMaximum(value);
				if (post_test != UINT32_MAX) return post_test;
				if (m_native_cfg.irreducible || !m_native_cfg.back_edges.empty() ||
				    inst->NumArgs() == 0u || (inst->Parent() != nullptr && InCycle(inst->Parent())))
					return PossibleU32Bits(value);
				uint32_t result = 0u;
				for (uint32_t arm = 0; arm < inst->NumArgs(); ++arm) {
					result = std::max(result, maximum(arm));
					if (result == UINT32_MAX) break;
				}
				return result;
			}
			case ValueOpcode::BitFieldUExtract: {
				uint32_t width;
				if (ImmediateU32(inst->Arg(2), width) && width < 32u) return (1u << width) - 1u;
				break;
			}
			case ValueOpcode::FindILsb32:
			case ValueOpcode::FindUMsb32:
				return (!guard.IsEmpty() && ImpliesNonzero(guard, inst->Arg(0))) ||
				               NonzeroOnEntry(inst->Arg(0), inst->Parent()) ? 31u : UINT32_MAX;
			case ValueOpcode::UMax32: return std::max(maximum(0), maximum(1));
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitFieldInsert: return PossibleU32Bits(value);
			case ValueOpcode::UMin32:
			case ValueOpcode::BitwiseAnd32: return std::min(maximum(0), maximum(1));
			case ValueOpcode::SMin32: {
				// Signed and unsigned minimums agree on nonnegative operands.
				const auto left = maximum(0), right = maximum(1);
				if (std::max(left, right) <= INT32_MAX) return std::min(left, right);
				break;
			}
			case ValueOpcode::ConvertS32F32: {
				// The conversion truncates toward zero.
				const auto [low, high] = FloatRange(inst->Arg(0), guard);
				if (low >= 0.0f && high < 0x1p31f) return static_cast<uint32_t>(high);
				break;
			}
			case ValueOpcode::SelectU32:
				if (!guard.IsEmpty() && Implies(guard, inst->Arg(0))) return maximum(1);
				return SelectorPredicateFalse(inst->Arg(0), bounds, depth + 1u, guard)
				           ? maximum(2) : std::max(maximum(1), maximum(2));
			case ValueOpcode::ISub32: {
				uint32_t left;
				if (ImmediateU32(inst->Arg(0), left) && maximum(1) <= left) return left;
				break;
			}
			case ValueOpcode::IAdd32: {
				const auto result = uint64_t {maximum(0)} + maximum(1);
				if (result <= UINT32_MAX) return static_cast<uint32_t>(result);
				break;
			}
			case ValueOpcode::IMul32: {
				const auto result = uint64_t {maximum(0)} * maximum(1);
				if (result <= UINT32_MAX) return static_cast<uint32_t>(result);
				break;
			}
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftLeftLogical32: {
				uint32_t shift;
				if (!ImmediateU32(inst->Arg(1), shift) || shift >= 32u) break;
				if (inst->GetOpcode() == ValueOpcode::ShiftRightLogical32) return maximum(0) >> shift;
				const auto result = uint64_t {maximum(0)} << shift;
				if (result <= UINT32_MAX) return static_cast<uint32_t>(result);
				break;
			}
			default: break;
		}
		return UINT32_MAX;
	}

	bool ImpossibleSwitchEdge(const Block* from, const Block* to) const {
		const auto& term = from->terminator;
		if (term.kind != CFG::TerminatorKind::IndirectBranch || !term.indexed) return false;
		const auto maximum = SelectorMaximum(from->indirect_target);
		bool found = false;
		for (const auto& branch: term.cases) {
			if (branch.target != to) continue;
			if (branch.value <= maximum) return false;
			found = true;
		}
		return found;
	}

	bool TryMakeFiniteDescriptor(Inst& handle, IndirectDescriptorPlan& plan) {
		const bool sampler = handle.GetOpcode() == ValueOpcode::GetSamplerResource;
		if ((!sampler && handle.GetOpcode() != ValueOpcode::GetImageResource) ||
		    handle.NumArgs() != (sampler ? 4u : 8u))
			return false;
		if (std::ranges::any_of(handle.Uses(), [&](const Use& use) {
			const auto op = use.user->GetOpcode();
			if (op != ValueOpcode::ImageSampleRaw) return sampler || op != ValueOpcode::ImageGatherRaw;
			return sampler && (m_program.memory_info[use.user->Flags<MemoryFlags>().index]
			                       .image_sample_flags & Decoder::ImageSampleFlagAdjust) != 0u;
		})) return false;
		// Normalize synchronized descriptor words to one GPU ordinal.
		// Build the complete graph before changing IR so an unsupported leaf is transactional.
		struct Choice {
			DescriptorSource descriptor;
			const Inst* branch = nullptr;
			std::vector<uint32_t> children;
			Value key;
		};
		std::vector<Choice> choices;
		struct VisitItem {
			uint32_t         parent_index = UINT32_MAX;
			uint32_t         child_slot   = 0u;
			DescriptorSource descriptor;
		};
		const auto run_visit = [&](DescriptorSource initial_descriptor) -> bool {
			std::vector<VisitItem> work {
			    {.parent_index = UINT32_MAX, .child_slot = 0u, .descriptor = std::move(initial_descriptor)}};
			while (!work.empty()) {
				auto item = std::move(work.back());
				work.pop_back();
				for (auto& word: item.descriptor.dwords) word = word.Resolve();
				uint32_t existing = UINT32_MAX;
				for (uint32_t i = 0; i < choices.size(); ++i) {
					if (choices[i].descriptor.dwords == item.descriptor.dwords) {
						existing = i;
						break;
					}
				}
				if (existing != UINT32_MAX) {
					if (item.parent_index != UINT32_MAX) {
						choices[item.parent_index].children[item.child_slot] = existing;
					}
					continue;
				}
				const auto index = static_cast<uint32_t>(choices.size());
				choices.push_back({std::move(item.descriptor)});
				if (item.parent_index != UINT32_MAX) {
					choices[item.parent_index].children[item.child_slot] = index;
				}
				const auto& descriptor = choices[index].descriptor;
				uint32_t    bad        = 0;
				if (ValidateSource(descriptor, bad)) {
					// Finite inline samplers have literal leaves; live descriptor tables use the
					// existing direct sampler planning path.
					if (sampler &&
					    !std::ranges::all_of(
					        std::span(descriptor.dwords).first(descriptor.dword_count),
					        [](Value word) { return word.IsImmediate() && word.GetType() == Type::U32; })) {
						return false;
					}
					continue;
				}
				const auto* branch = descriptor.dwords[bad].TryInstruction();
				if (branch == nullptr || branch->Parent() == nullptr) return false;
				const auto op   = branch->GetOpcode();
				const bool phi  = op == ValueOpcode::Phi;
				const bool lane = op == ValueOpcode::ReadFirstLane;
				if ((!phi && !lane && op != ValueOpcode::SelectU32) ||
				    (phi && branch->NumPhiBlocks() != branch->NumArgs())) {
					return false;
				}
				if (lane && !std::ranges::all_of(handle.Uses(), [&](const Use& use) {
					    return Implies(use.user->Arg(use.user->NumArgs() - 1u), branch->Arg(1));
				    })) {
					return false;
				}
				const auto count = phi ? branch->NumArgs() : lane ? 1u : 2u;
				std::array<const Inst*, 8> selected_words {};
				const auto*                anchor = branch;
				for (uint32_t word = 0; word < descriptor.dword_count; ++word) {
					const auto* selected = descriptor.dwords[word].TryInstruction();
					if (selected == nullptr || selected->GetOpcode() != op ||
					    selected->Parent() != branch->Parent()) {
						if (!ValidateRuntimeValue(m_program, descriptor.dwords[word])) return false;
						continue;
					}
					if (phi ? selected->NumArgs() != count || selected->NumPhiBlocks() != count
					        : !EquivalentValue(m_program, selected->Arg(lane ? 1u : 0u),
					                           branch->Arg(lane ? 1u : 0u))) {
						return false;
					}
					selected_words[word] = selected;
					if (!phi && AvailableBefore(*anchor, *selected)) anchor = selected;
				}
				// Later words may depend on choices computed after the first descriptor word.
				choices[index].branch = anchor;
				choices[index].children.assign(count, UINT32_MAX);
				std::vector<VisitItem> children_to_push;
				children_to_push.reserve(count);
				for (uint32_t arm = 0; arm < count; ++arm) {
					// A nonzero bit scan cannot choose its zero-input switch sentinel.
					// Preserve the Phi edge; its ordinal is never observed at runtime.
					if (phi && ImpossibleSwitchEdge(anchor->PhiBlock(arm), anchor->Parent())) {
						continue;
					}
					auto child = descriptor;
					for (uint32_t word = 0; word < child.dword_count; ++word) {
						const auto* selected = selected_words[word];
						if (selected == nullptr) continue;
						uint32_t incoming = lane ? 0u : arm + 1u;
						if (phi) {
							incoming = 0u;
							while (incoming < count &&
							       selected->PhiBlock(incoming) != anchor->PhiBlock(arm)) {
								++incoming;
							}
							if (incoming == count) return false;
						}
						child.dwords[word] = selected->Arg(incoming);
					}
					children_to_push.push_back(
					    {.parent_index = index, .child_slot = arm, .descriptor = std::move(child)});
				}
				for (size_t i = children_to_push.size(); i-- > 0;) {
					work.push_back(std::move(children_to_push[i]));
				}
			}
			return true;
		};
		DescriptorSource root;
		root.dword_count = handle.NumArgs();
		for (uint32_t word = 0; word < root.dword_count; ++word) root.dwords[word] = handle.Arg(word);
		if (!run_visit(std::move(root)) || choices.front().branch == nullptr) return false;
		DescriptorSource finite_source;
		finite_source.dword_count = root.dword_count;
		finite_source.dwords.fill(Value(0u));
		finite_source.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{});
		auto& sources = finite_source.indirect_descriptor->sources;
		for (auto& choice: choices) {
			if (choice.branch != nullptr) continue;
			const auto source = InternSource(choice.descriptor);
			auto found = std::ranges::find(sources, source);
			choice.key = Value(static_cast<uint32_t>(found - sources.begin()));
			if (found == sources.end()) sources.push_back(source);
		}
		if (sources.empty()) return false;
		for (auto& choice: choices) {
			if (choice.branch == nullptr) continue;
			auto* block = choice.branch->Parent();
			const auto where = std::ranges::find_if(block->Instructions(), [&](const Inst& inst) {
				return &inst == choice.branch;
			});
			const auto op = choice.branch->GetOpcode();
			auto& key = *(op == ValueOpcode::Phi ? block->PrependNewInst(where, op)
			    : op == ValueOpcode::ReadFirstLane
			        ? block->PrependNewInst(where, op, {Value(0u), choice.branch->Arg(1)})
			        : block->PrependNewInst(where, op, {choice.branch->Arg(0), Value(0u), Value(0u)}));
			if (op == ValueOpcode::Phi) key.SetFlags(Type::U32);
			choice.key = Value(&key);
		}
		for (const auto& choice: choices) {
			if (choice.branch == nullptr) continue;
			auto* key = choice.key.Instruction();
			const bool phi = choice.branch->GetOpcode() == ValueOpcode::Phi;
			const bool lane = choice.branch->GetOpcode() == ValueOpcode::ReadFirstLane;
			for (uint32_t arm = 0; arm < choice.children.size(); ++arm) {
				const auto child = choice.children[arm];
				const auto value = child == UINT32_MAX ? Value(0u) : choices[child].key;
				if (phi) key->AddPhiOperand(choice.branch->PhiBlock(arm), value);
				else key->SetArg(lane ? 0u : arm + 1u, value);
			}
		}
		plan.handle = &handle;
		plan.source = InternSource(finite_source);
		plan.key = choices.front().key;
		plan.roots = finite_source.dwords;
		plan.reads.fill(nullptr);
		return true;
	}

	const IndirectDescriptorPlan* FindIndirectDescriptor(const Inst& handle) const {
		const auto found =
		    std::find_if(m_indirect_descriptors.begin(), m_indirect_descriptors.end(),
		                 [&](const IndirectDescriptorPlan& plan) {
			    return plan.handle == &handle;
		    });
		return found == m_indirect_descriptors.end() ? nullptr : &*found;
	}

	bool IsIndirectPlanningMemory(uint32_t index) const {
		return std::any_of(m_indirect_descriptors.begin(), m_indirect_descriptors.end(),
		                   [&](const IndirectDescriptorPlan& plan) {
			return plan.reads[0] != nullptr && !plan.retain_reads &&
			       std::find(plan.memory.begin(), plan.memory.begin() + plan.read_dwords, index) !=
			           plan.memory.begin() + plan.read_dwords;
		});
	}

	void PlanIndirectDescriptors() {
		// Earlier proofs predate rewrites and erasures; planning only adds instructions.
		m_loop_proofs.clear();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				if ((ImageOpcodeInfoOf(inst.GetOpcode()).access == ImageAccess::None &&
				     !IndirectBufferAccess(inst)) ||
				    inst.NumArgs() == 0u) {
					continue;
				}
				const auto flags = inst.Flags<MemoryFlags>();
				const auto& memory = m_program.memory_info[flags.index];
				const auto plan_handle = [&](Value value, uint32_t width, uint32_t base, bool sampler) {
					auto* handle = value.Resolve().TryInstruction();
					if (handle == nullptr || FindIndirectDescriptor(*handle) != nullptr) return;
					DescriptorSource descriptor;
					MakeSource(*handle, width, sampler, false, base, descriptor, flags.pc);
					uint32_t bad_dword = 0;
					if (ValidateSource(descriptor, bad_dword)) return;
					IndirectDescriptorPlan plan;
					plan.read_dwords = width;
					if (!sampler && width == 8u && memory.image_r128) {
						// R128 handles are normalized to eight words; only their low four
						// words come from memory. Every use must agree on that encoding.
						if (!std::ranges::all_of(std::span(descriptor.dwords).subspan(4u),
						        [](Value word) { return word.Resolve() == Value(0u); }) ||
						    !std::ranges::all_of(handle->Uses(), [&](const Use& use) {
							    return ImageOpcodeInfoOf(use.user->GetOpcode()).access != ImageAccess::None &&
							        m_program.memory_info[use.user->Flags<MemoryFlags>().index].image_r128;
						    })) return;
						plan.read_dwords = 4u;
					}
					if (TryMakeNestedDescriptor(*handle, descriptor, plan) ||
					    (sampler && TryMakeFiniteDescriptor(*handle, plan)) ||
					    TryMakeTableDescriptor(*handle, descriptor, plan) ||
					    (!sampler && (TryMakeFiniteDescriptor(*handle, plan) ||
					                  TryMakeIndirectBuffer(*handle, descriptor, plan)))) {
						for (const auto& previous: m_indirect_descriptors) {
							if (plan.reads[0] != nullptr && previous.reads[0] == plan.reads[0] &&
							    !EquivalentValue(m_program, previous.key, plan.key))
								Fail(flags.pc, "shared descriptor read has incompatible keys");
						}
						m_indirect_descriptors.push_back(std::move(plan));
					}
				};
				if (inst.GetOpcode() == ValueOpcode::ImageSampleRaw &&
				    (memory.image_sample_flags & Decoder::ImageSampleFlagAdjust) == 0u) {
					plan_handle(inst.Arg(1), 4u, memory.sampler * 4u, true);
				}
				plan_handle(inst.Arg(0), IndirectBufferAccess(inst) ? 4u : 8u,
				            memory.resource * 4u, false);
			}
		}
	}

	bool GetHandle(Value value, ValueOpcode expected, uint32_t width, uint32_t pc,
	               uint32_t base_reg, Inst*& handle, uint32_t& source, bool sampler = false,
	               bool sample_adjust = false) {
		handle = value.Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != expected) {
			Fail(pc, fmt::format("memory operation requires {}", ValueOpcodeName(expected)));
		}
		DescriptorSource descriptor;
		MakeSource(*handle, width, sampler, sample_adjust, base_reg, descriptor, pc);
		uint32_t bad_dword = 0;
		if (!ValidateSource(descriptor, bad_dword)) {
			if (expected == ValueOpcode::GetBufferResource &&
			    std::all_of(descriptor.dwords.begin(), descriptor.dwords.begin() + width,
			                [](Value word) { return word.Resolve().GetType() == Type::U32; })) {
				return false;
			}
			Fail(pc, fmt::format("{} dword {} is not a valid runtime value",
			                     ValueOpcodeName(expected), bad_dword));
		}
		source = InternSource(descriptor);
		return true;
	}

	void ValidateAddressHandle(Value value, uint32_t pc) const {
		const auto* handle = value.Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource) {
			Fail(pc, "address operation requires GetAddressResource");
		}
		if (handle->NumArgs() != 2) {
			Fail(pc, "GetAddressResource must have two address dwords");
		}
	}

	// Returns the source of the one bound buffer with a fixed V# that a write's Phi web carries
	// word by word, else UINT32_MAX. The web's other definitions are taken to reach the write only
	// with EXEC zero, as in a loop-exit undo of an in-loop atomic.
	uint32_t CarriedBufferSource(const Inst& handle) const {
		std::array<std::vector<Value>, 4> webs;
		for (uint32_t word = 0; word < 4u; ++word)
			webs[word] = PhiWeb(handle.Arg(word));
		const auto carries = [&](const BufferResource& buffer) {
			if (m_sources[buffer.source].indirect_descriptor) return false;
			for (uint32_t word = 0; word < 4u; ++word) {
				const auto carried = [&](Value node) {
					return EquivalentValue(m_program, node, m_sources[buffer.source].dwords[word]);
				};
				if (std::ranges::none_of(webs[word], carried)) return false;
			}
			return true;
		};
		const auto bound = std::ranges::find_if(m_info.buffers, carries);
		if (bound == m_info.buffers.end() ||
		    std::ranges::any_of(std::next(bound), m_info.buffers.end(), carries))
			return UINT32_MAX;
		return bound->source;
	}

	uint32_t AddBuffer(uint32_t source, const MemoryInfo& memory, ValueOpcode op, uint32_t pc) {
		for (uint32_t i = 0; i < m_info.buffers.size(); i++) {
			if (m_info.buffers[i].source == source) {
				Merge(m_info.buffers[i], memory, op, pc);
				return i;
			}
		}
		if (m_info.buffers.size() >= ShaderInfo::MaxBufferRoots) {
			return UINT32_MAX;
		}
		BufferResource resource;
		resource.source       = source;
		resource.first_use_pc = pc;
		Merge(resource, memory, op, pc);
		m_info.buffers.push_back(resource);
		return static_cast<uint32_t>(m_info.buffers.size() - 1);
	}

	static void Merge(BufferResource& resource, const MemoryInfo& memory, ValueOpcode op,
	                  uint32_t pc) {
		const auto access        = BufferAccessOf(op);
		const bool atomic        = access == BufferAccess::Atomic;
		const bool write         = access == BufferAccess::Write || atomic;
		resource.first_use_pc    = std::min(resource.first_use_pc, pc);
		resource.max_byte_extent = std::max(resource.max_byte_extent, ByteExtent(memory));
		resource.read            = resource.read || !write || atomic;
		resource.written         = resource.written || write;
		resource.atomic          = resource.atomic || atomic;
		resource.formatted       = resource.formatted || memory.formatted;
		resource.scalar          = resource.scalar || op == ValueOpcode::ReadConstBuffer ||
		                           memory.kind == ResourceKind::ScalarBuffer;
	}

	uint32_t AddImage(uint32_t source, const MemoryInfo& memory, ValueOpcode op, uint32_t pc) {
		const auto resource_class = ImageOpcodeInfoOf(op).resource_class;
		const bool atomic64 = ImageOpcodeInfoOf(op).access == ImageAccess::Atomic &&
		                      memory.data_bits == 64u;
		const bool dynamic_mip =
		    (resource_class == ImageResourceClass::Storage && memory.image_has_mip) ||
		    (op == ValueOpcode::ImageGatherRaw &&
		     (memory.image_sample_flags & Decoder::ImageSampleFlagLod) != 0u);
		const auto mip = dynamic_mip ? ImageMipMode::Dynamic : ImageMipMode::None;
		const bool depth = (memory.image_sample_flags & Decoder::ImageSampleFlagCompare) != 0;
		for (uint32_t i = 0; i < m_info.images.size(); i++) {
			auto& image = m_info.images[i];
			if (image.source == source && image.resource_class == resource_class &&
			    image.dimension == memory.image_dimension && image.mip_mode == mip &&
			    image.depth_compare == depth && image.r128 == memory.image_r128 &&
			    image.atomic64 == atomic64) {
				image.cube |= memory.image_cube;
				Merge(image, op, pc);
				return i;
			}
		}
		if (m_info.images.size() >= ShaderInfo::MaxImages) {
			return UINT32_MAX;
		}
		ImageResource image;
		image.source         = source;
		image.first_use_pc   = pc;
		image.resource_class = resource_class;
		image.dimension      = memory.image_dimension;
		image.mip_mode       = mip;
		image.depth_compare  = depth;
		image.r128           = memory.image_r128;
		image.cube           = memory.image_cube;
		image.atomic64       = atomic64;
		Merge(image, op, pc);
		m_info.images.push_back(image);
		return static_cast<uint32_t>(m_info.images.size() - 1);
	}

	static void Merge(ImageResource& image, ValueOpcode op, uint32_t pc) {
		const auto access  = ImageOpcodeInfoOf(op).access;
		const bool atomic  = access == ImageAccess::Atomic;
		const bool write   = access == ImageAccess::Write || atomic;
		image.first_use_pc = std::min(image.first_use_pc, pc);
		image.read         = image.read || !write || atomic;
		image.written      = image.written || write;
		image.atomic       = image.atomic || atomic;
		image.unfiltered_access_only &= op == ValueOpcode::ImageRead || op == ValueOpcode::ImageWrite;
		image.sample_access_only &= op == ValueOpcode::ImageSampleRaw;
	}

	uint32_t AddSampler(uint32_t source, uint32_t pc) {
		for (uint32_t i = 0; i < m_info.samplers.size(); i++) {
			if (m_info.samplers[i].source == source) {
				m_info.samplers[i].first_use_pc = std::min(m_info.samplers[i].first_use_pc, pc);
				return i;
			}
		}
		if (m_info.samplers.size() >= ShaderInfo::MaxSamplers) {
			return UINT32_MAX;
		}
		m_info.samplers.push_back({source, pc});
		const auto root = static_cast<uint32_t>(m_info.samplers.size() - 1);
		const auto* descriptor = Source(source);
		if (descriptor != nullptr && descriptor->indirect_descriptor) {
			const auto& sources = descriptor->indirect_descriptor->sources;
			if (sources.empty()) return root;
			std::vector<uint32_t> resources {root};
			for (size_t i = 1; i < sources.size(); ++i) {
				const auto candidate = AddSampler(sources[i], pc);
				if (candidate == UINT32_MAX) return UINT32_MAX;
				resources.push_back(candidate);
			}
			m_info.samplers[root].indirect_resources = std::move(resources);
		}
		return root;
	}

	void AddSampledPair(uint32_t image, uint32_t sampler, uint32_t pc) {
		for (auto& pair: m_info.sampled_pairs) {
			if (pair.image == image && pair.sampler == sampler) {
				pair.first_use_pc = std::min(pair.first_use_pc, pc);
				return;
			}
		}
		m_info.sampled_pairs.push_back({image, sampler, pc});
	}

	void AddHandlePatch(Inst* handle, uint32_t resource, uint32_t pc) {
		for (const auto& patch: m_handle_patches) {
			if (patch.handle == handle) {
				if (patch.resource != resource) {
					Fail(pc, fmt::format("{} is reused with incompatible resource classes",
					                     ValueOpcodeName(handle->GetOpcode())));
				}
				return;
			}
		}
		m_handle_patches.push_back({handle, resource});
	}

	void AddMemoryPatch(uint32_t index, uint32_t resource, uint32_t sampler, bool has_sampler,
	                    uint32_t pc) {
		for (auto& patch: m_memory_patches) {
			if (patch.index != index) {
				continue;
			}
			if (patch.resource != resource ||
			    (has_sampler && patch.has_sampler && patch.sampler != sampler)) {
				Fail(pc, "memory metadata is reused with incompatible resources");
			}
			if (has_sampler) {
				patch.sampler     = sampler;
				patch.has_sampler = true;
			}
			return;
		}
		m_memory_patches.push_back({index, resource, sampler, has_sampler});
	}

	void Collect(Inst& inst) {
		const auto op           = inst.GetOpcode();
		const auto buffer       = BufferAccessOf(op);
		const auto address_info = AddressOpcodeInfoOf(op);
		const auto image_info   = ImageOpcodeInfoOf(op);
		if (buffer == BufferAccess::None && address_info.access == AddressAccess::None &&
		    image_info.access == ImageAccess::None) {
			return;
		}
		const auto flags = inst.Flags<MemoryFlags>();
		if (flags.index >= m_program.memory_info.size()) {
			Fail(flags.pc, fmt::format("memory metadata index {} is out of range", flags.index));
		}
		if (inst.NumArgs() == 0) {
			Fail(flags.pc, "memory operation has no resource handle");
		}
		const auto& memory = m_program.memory_info[flags.index];
		if (memory.planning_only || IsIndirectPlanningMemory(flags.index)) {
			return;
		}
		Inst*    handle   = nullptr;
		uint32_t source   = 0;
		uint32_t resource = 0;

		if (buffer != BufferAccess::None) {
			handle = inst.Arg(0).Resolve().TryInstruction();
			const auto* indirect = handle == nullptr ? nullptr : FindIndirectDescriptor(*handle);
			if (indirect != nullptr) {
				source = indirect->source;
			} else if (!GetHandle(inst.Arg(0), ValueOpcode::GetBufferResource, 4, flags.pc,
			               memory.resource * 4u, handle, source)) {
				if (memory.kind == (op == ValueOpcode::ReadConstBuffer ? ResourceKind::ScalarBuffer
				                                                       : ResourceKind::Buffer) &&
				    memory.SupportsIndirectBufferLoad(op)) {
					m_program.memory_info[flags.index].kind = ResourceKind::IndirectBuffer;
					return;
				}
				source = buffer != BufferAccess::Read ? CarriedBufferSource(*handle) : UINT32_MAX;
				if (source == UINT32_MAX) {
					Fail(flags.pc,
					     "buffer descriptor is not a valid runtime value; GPU-selected access "
					     "requires a scalar, raw 8/16-bit, raw DWORD x1/x2/x3/x4, or formatted X load");
				}
			}
			resource = AddBuffer(source, memory, op, flags.pc);
			if (resource == UINT32_MAX) {
				Fail(flags.pc, "buffer resource limit exceeded");
			}
			AddHandlePatch(handle, resource, flags.pc);
			AddMemoryPatch(flags.index, resource, 0, false, flags.pc);
			return;
		}
		if (address_info.access != AddressAccess::None) {
			if (!IsAddressResourceKind(memory.kind)) {
				Fail(flags.pc, "address operation has invalid resource kind");
			}
			const auto active = inst.Arg(inst.NumArgs() - 1);
			if (memory.kind == ResourceKind::Flat && memory.address_is_full &&
			    IsNoncanonicalFlatAddress(inst.Arg(2), active)) {
				ValidateAddressHandle(inst.Arg(0), flags.pc);
				if (memory.data_bits != 32u || m_program.scratch_dwords == 0) {
					Fail(flags.pc, "local FLAT access requires DWORD data and per-thread scratch storage");
				}
				m_program.memory_info[flags.index].kind = ResourceKind::FlatLocal;
				return;
			}
			if (memory.kind == ResourceKind::Scratch) {
				handle = inst.Arg(0).Resolve().TryInstruction();
				if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetScratchResource ||
				    handle->NumArgs() != 0) {
					Fail(flags.pc, "scratch operation requires GetScratchResource");
				}
				if (m_program.scratch_dwords == 0) {
					Fail(flags.pc, "scratch operation requires a nonzero per-thread scratch size");
				}
				return;
			}
			ValidateAddressHandle(inst.Arg(0), flags.pc);
			if (address_info.access == AddressAccess::Write) {
				m_program.has_address_writes = true;
			}
			handle           = inst.Arg(0).Resolve().TryInstruction();
			const auto* low  = AddressBaseRead(handle->Arg(0), active);
			const auto* high = AddressBaseRead(handle->Arg(1), active);
			if (low == nullptr || high == nullptr || low->GetOpcode() != high->GetOpcode()) return;
			const bool user = low->GetOpcode() == ValueOpcode::GetUserData;
			const auto slot = [&](const Inst* read) {
				return user ? RegIndex(read->Arg(0).ScalarRegister()) - m_program.user_data_base
				            : read->Arg(1).Resolve().U32();
			};
			const std::array base {slot(low), slot(high)};
			auto&            bases = user ? m_info.user_address_bases : m_info.address_bases;
			if (std::ranges::find(bases, base) == bases.end()) bases.push_back(base);
			return;
		}

		if (memory.kind != ResourceKind::Image ||
		    image_info.resource_class == ImageResourceClass::None) {
			Fail(flags.pc, "image operation has invalid resource kind");
		}
		handle               = inst.Arg(0).Resolve().TryInstruction();
		const auto* indirect = handle != nullptr ? FindIndirectDescriptor(*handle) : nullptr;
		if (indirect != nullptr) {
			source = indirect->source;
		} else {
			GetHandle(inst.Arg(0), ValueOpcode::GetImageResource, 8, flags.pc,
			          memory.resource * 4u, handle, source);
		}
		resource = AddImage(source, memory, op, flags.pc);
		if (resource == UINT32_MAX) {
			Fail(flags.pc, "image resource limit exceeded");
		}
		AddHandlePatch(handle, resource, flags.pc);
		uint32_t sampler = 0;
		if (image_info.needs_sampler) {
			if (inst.NumArgs() < 2) {
				Fail(flags.pc, "sampled image operation has no sampler handle");
			}
			Inst*      sampler_handle = nullptr;
			uint32_t   sampler_source = 0;
			const bool sample_adjust =
			    (memory.image_sample_flags & Decoder::ImageSampleFlagAdjust) != 0;
			sampler_handle = inst.Arg(1).Resolve().TryInstruction();
			const auto* selected_sampler = sampler_handle != nullptr
			    ? FindIndirectDescriptor(*sampler_handle) : nullptr;
			if (selected_sampler != nullptr) {
				sampler_source = selected_sampler->source;
			} else {
				GetHandle(inst.Arg(1), ValueOpcode::GetSamplerResource, 4, flags.pc,
				          memory.sampler * 4u, sampler_handle, sampler_source, true, sample_adjust);
			}
			sampler = AddSampler(sampler_source, flags.pc);
			if (sampler == UINT32_MAX) {
				Fail(flags.pc, "sampler resource limit exceeded");
			}
			m_info.samplers[sampler].gather_lod |=
			    op == ValueOpcode::ImageGatherRaw &&
			    (memory.image_sample_flags & Decoder::ImageSampleFlagLod) != 0u;
			AddHandlePatch(sampler_handle, sampler, flags.pc);
			AddSampledPair(resource, sampler, flags.pc);
		}
		AddMemoryPatch(flags.index, resource, sampler, image_info.needs_sampler, flags.pc);
	}

	const DescriptorSource* Source(uint32_t source) const {
		return source < m_sources.size() ? &m_sources[source] : nullptr;
	}

	Program&                                   m_program;
	const Decoder::Program&                    m_decoded;
	const CFG::Graph&                          m_native_cfg;
	ShaderStageInputInfo                       m_input;
	std::vector<Program::ScalarWrite>          m_scalar_writes;
	std::vector<ResolvedHandle>                m_resolved_handles;
	std::vector<ScalarReadStep>                m_srt_walk;
	std::unordered_set<const Inst*>            m_srt_visited;
	std::vector<MemoryIndexUse>                m_memory_index_uses;
	std::vector<Inst*>                         m_scalar_reads;
	ShaderInfo                                 m_info;
	std::vector<DescriptorSource>              m_sources;
	std::vector<HandlePatch>                   m_handle_patches;
	std::vector<MemoryPatch>                   m_memory_patches;
	std::vector<IndirectDescriptorPlan>             m_indirect_descriptors;
	std::map<const Inst*, Value>              m_descriptor_values;
	mutable std::unordered_map<const Inst*, bool>   m_loop_proofs; // By loop-bound comparison.
	bool                                       m_shader_writes      = false;
	bool                                       m_has_address_writes = false;
};

UniformFillPlan RecognizePatternFill(Program& program, const Decoder::Program& decoded,
                                   ShaderStageInputInfo input) {
	struct Pattern {
		size_t code_words;
		uint64_t hash;
		uint32_t wave_size_mask;
		bool bounded;
	};
	constexpr std::array patterns {
		// Bounded 64-lane variant.
		Pattern {.code_words = 69, .hash = 0x0704b9818e3cba06ull, .wave_size_mask = 64, .bounded = true},
		// Dispatch thread dimensions, 64-lane variants.
		Pattern {.code_words = 42, .hash = 0x92ce57f21310ef02ull, .wave_size_mask = 64, .bounded = false},
		Pattern {.code_words = 40, .hash = 0x374161027f58bee7ull, .wave_size_mask = 64, .bounded = false},
		Pattern {.code_words = 40, .hash = 0x7d7172c56c2e296dull, .wave_size_mask = 64, .bounded = false},
		// Dispatch thread dimensions, valid in both wave sizes.
		Pattern {.code_words = 38, .hash = 0xacbda1a1ccdd4198ull, .wave_size_mask = 32 | 64, .bounded = false},
	};
	if (program.stage != ShaderType::Compute || decoded.instructions.empty() ||
	    program.user_data_base != 0 || program.user_data_count < 10 ||
	    program.info.buffers.size() != 1 || !program.info.images.empty()) return {};
	const auto& cs = *input.compute;
	const bool matching_inputs = cs.workgroup_register == 10 && cs.thread_ids_num == 1 &&
	    cs.group_id[0] && !cs.group_id[1] && !cs.group_id[2] && cs.threads_num[0] == 64 &&
	    cs.threads_num[1] == 1 && cs.threads_num[2] == 1 && cs.float_mode == 0xc0;
	if (!matching_inputs) return {};
	const auto& last = decoded.instructions.back();
	const size_t words = last.pc / sizeof(uint32_t) + last.word_count;
	if (words > decoded.code.size() ||
	    std::ranges::find(patterns, words, &Pattern::code_words) == patterns.end()) return {};
	// Hash executable instructions through S_ENDPGM, excluding binary metadata.
	// Verified variants implement Destination[id] = value[id % pattern_size].
	const auto hash = XXH3_64bits(decoded.code.data(), words * sizeof(uint32_t));
	const auto pattern = std::ranges::find_if(patterns, [&](const Pattern& candidate) {
		return candidate.code_words == words && candidate.hash == hash;
	});
	if (pattern == patterns.end() || (program.wave_size & pattern->wave_size_mask) == 0 ||
	    cs.dispatch_thread_dimensions != !pattern->bounded) return {};
	std::array<Value, 10> userdata;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::GetUserData) continue;
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < userdata.size()) userdata[reg] = Value(&inst);
		}
	}
	for (uint32_t reg = 4; reg < 10; ++reg)
		if ((reg != 8 || pattern->bounded) && userdata[reg].IsEmpty()) return {};
	UniformFillPlan result;
	result.fill.kind = UniformFillKind::Buffer;
	result.fill.words = 1;
	result.fill.group_stride[0] = 64;
	std::copy_n(userdata.begin() + 4, 4, result.values.begin());
	result.pattern_size = userdata[9];
	// Only the bounded variant reads an explicit count; the other uses thread dimensions.
	if (pattern->bounded) result.element_count = userdata[8];
	return result;
}

} // namespace

void TrackResources(Program& program, const Decoder::Program& decoded, const CFG::Graph& native_cfg,
                    ShaderStageInputInfo input) {
	Tracker(program, decoded, native_cfg, input).Run();
	program.uniform_fill = RecognizePatternFill(program, decoded, input);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
