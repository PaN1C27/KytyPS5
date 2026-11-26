#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

struct ScalarCond {
	Value                               predicate;
	std::optional<CFG::BranchCondition> vote_kind;
	bool                                negated = false;
	bool                                valid   = false;
};

struct DominatedFact {
	Value                               predicate;
	std::optional<CFG::BranchCondition> vote_kind;
	bool                                truth = false;
};

ScalarCond UnwrapScalarCondition(Value cond) {
	ScalarCond result;
	cond = cond.Resolve();
	if (cond.IsEmpty() || cond.GetType() != Type::U1) {
		return result;
	}
	bool negated = false;
	while (const auto* inst = cond.TryInstruction()) {
		if (inst->GetOpcode() != ValueOpcode::LogicalNot) {
			break;
		}
		negated = !negated;
		cond    = inst->Arg(0).Resolve();
	}
	if (const auto* inst = cond.TryInstruction();
	    inst != nullptr && inst->GetOpcode() == ValueOpcode::ConditionRef) {
		const auto kind = inst->Flags<CFG::BranchCondition>();
		if (kind != CFG::BranchCondition::SccZero && kind != CFG::BranchCondition::SccNonZero &&
		    kind != CFG::BranchCondition::ScalarInstruction) {
			return result;
		}
		result.vote_kind = kind;
		cond             = inst->Arg(0).Resolve();
	}
	if (cond.IsEmpty() || cond.GetType() != Type::U1) {
		return result;
	}
	result.predicate = cond;
	result.negated   = negated;
	result.valid     = true;
	return result;
}

bool IsPureAluOpcode(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IAbs32:
		case ValueOpcode::IMul32:
		case ValueOpcode::UMulHi:
		case ValueOpcode::SMulHi:
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
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::BitReverse32:
		case ValueOpcode::BitCount32:
		case ValueOpcode::FindILsb32:
		case ValueOpcode::FindUMsb32:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::SMin32:
		case ValueOpcode::SMax32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::UGreaterThanEqual32:
		case ValueOpcode::SLessThan32:
		case ValueOpcode::SLessThanEqual32:
		case ValueOpcode::SGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::CompositeConstructU32x3:
		case ValueOpcode::CompositeExtractU32x3:
		case ValueOpcode::CompositeConstructU32x4:
		case ValueOpcode::CompositeExtractU32x4: return true;
		default: return false;
	}
}

bool IsCommutativeOpcode(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::IEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IMul32:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::SMin32:
		case ValueOpcode::SMax32: return true;
		default: return false;
	}
}

bool PureValuesEqual(Value a, Value b, uint32_t depth = 0) {
	a = a.Resolve();
	b = b.Resolve();
	if (a == b) {
		return true;
	}
	if (depth >= 8u || a.IsImmediate() || b.IsImmediate() || a.GetType() != b.GetType()) {
		return false;
	}
	const auto* lhs = a.TryInstruction();
	const auto* rhs = b.TryInstruction();
	if (lhs == nullptr || rhs == nullptr) {
		return false;
	}
	const auto op = lhs->GetOpcode();
	if (rhs->GetOpcode() != op || !IsPureAluOpcode(op) || lhs->NumArgs() != rhs->NumArgs() ||
	    lhs->Flags<uint64_t>() != rhs->Flags<uint64_t>()) {
		return false;
	}
	bool in_order = true;
	for (size_t i = 0; i < lhs->NumArgs(); ++i) {
		if (!PureValuesEqual(lhs->Arg(i), rhs->Arg(i), depth + 1u)) {
			in_order = false;
			break;
		}
	}
	if (in_order) {
		return true;
	}
	if (lhs->NumArgs() == 2u && IsCommutativeOpcode(op)) {
		return PureValuesEqual(lhs->Arg(0), rhs->Arg(1), depth + 1u) &&
		       PureValuesEqual(lhs->Arg(1), rhs->Arg(0), depth + 1u);
	}
	return false;
}

bool IsSubgroupUniformExpr(Value v, uint32_t depth = 0) {
	v = v.Resolve();
	if (v.IsEmpty()) {
		return false;
	}
	if (v.IsImmediate()) {
		return true;
	}
	if (depth >= 12u) {
		return false;
	}
	const auto* inst = v.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::GetUserData:
		case ValueOpcode::GetShaderBase:
		case ValueOpcode::ReadConst:
		case ValueOpcode::ReadFirstLane:
		case ValueOpcode::ReadLane:
		case ValueOpcode::Ballot:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer: return true;
		case ValueOpcode::GetBuiltin: {
			if (inst->NumArgs() < 1u) {
				return false;
			}
			const auto kind = inst->Arg(0).Resolve();
			return kind.IsImmediate() && kind.GetType() == Type::U32 &&
			       (kind.U32() == static_cast<uint32_t>(StageInputKind::WorkgroupId) ||
			        kind.U32() == static_cast<uint32_t>(StageInputKind::NumWorkgroups));
		}
		default: break;
	}
	if (!IsPureAluOpcode(inst->GetOpcode())) {
		return false;
	}
	for (size_t i = 0; i < inst->NumArgs(); ++i) {
		if (!IsSubgroupUniformExpr(inst->Arg(i), depth + 1u)) {
			return false;
		}
	}
	return true;
}

enum class CanonicalCmpKind { Eq32, Ne32, Eq64, Ne64, ULt32, ULe32, SLt32, SLe32 };

struct CanonicalCmp {
	CanonicalCmpKind kind;
	Value            first;
	Value            second;
};

std::optional<CanonicalCmp> CanonicalizeCmp(Value v) {
	const auto* inst = v.Resolve().TryInstruction();
	if (inst == nullptr || inst->NumArgs() != 2u) {
		return std::nullopt;
	}
	const Value a = inst->Arg(0).Resolve();
	const Value b = inst->Arg(1).Resolve();
	switch (inst->GetOpcode()) {
		case ValueOpcode::IEqual32: return CanonicalCmp {CanonicalCmpKind::Eq32, a, b};
		case ValueOpcode::INotEqual32: return CanonicalCmp {CanonicalCmpKind::Ne32, a, b};
		case ValueOpcode::IEqual64: return CanonicalCmp {CanonicalCmpKind::Eq64, a, b};
		case ValueOpcode::INotEqual64: return CanonicalCmp {CanonicalCmpKind::Ne64, a, b};
		case ValueOpcode::ULessThan32: return CanonicalCmp {CanonicalCmpKind::ULt32, a, b};
		case ValueOpcode::ULessThanEqual32: return CanonicalCmp {CanonicalCmpKind::ULe32, a, b};
		case ValueOpcode::UGreaterThan32: return CanonicalCmp {CanonicalCmpKind::ULt32, b, a};
		case ValueOpcode::UGreaterThanEqual32: return CanonicalCmp {CanonicalCmpKind::ULe32, b, a};
		case ValueOpcode::SLessThan32: return CanonicalCmp {CanonicalCmpKind::SLt32, a, b};
		case ValueOpcode::SLessThanEqual32: return CanonicalCmp {CanonicalCmpKind::SLe32, a, b};
		case ValueOpcode::SGreaterThan32: return CanonicalCmp {CanonicalCmpKind::SLt32, b, a};
		case ValueOpcode::SGreaterThanEqual32: return CanonicalCmp {CanonicalCmpKind::SLe32, b, a};
		default: return std::nullopt;
	}
}

std::optional<bool> ComparePredicates(Value p1, Value p2) {
	bool flip = false;
	p1        = p1.Resolve();
	while (const auto* inst = p1.TryInstruction()) {
		if (inst->GetOpcode() != ValueOpcode::LogicalNot) {
			break;
		}
		flip = !flip;
		p1   = inst->Arg(0).Resolve();
	}
	p2 = p2.Resolve();
	while (const auto* inst = p2.TryInstruction()) {
		if (inst->GetOpcode() != ValueOpcode::LogicalNot) {
			break;
		}
		flip = !flip;
		p2   = inst->Arg(0).Resolve();
	}
	if (PureValuesEqual(p1, p2)) {
		return !flip;
	}
	const auto c1 = CanonicalizeCmp(p1);
	const auto c2 = CanonicalizeCmp(p2);
	if (!c1 || !c2) {
		return std::nullopt;
	}
	const auto eq_group = [](CanonicalCmpKind k) {
		if (k == CanonicalCmpKind::Eq32 || k == CanonicalCmpKind::Ne32) return 1;
		if (k == CanonicalCmpKind::Eq64 || k == CanonicalCmpKind::Ne64) return 2;
		return 0;
	};
	if (eq_group(c1->kind) != 0 && eq_group(c1->kind) == eq_group(c2->kind)) {
		const bool same_ops =
		    (PureValuesEqual(c1->first, c2->first) && PureValuesEqual(c1->second, c2->second)) ||
		    (PureValuesEqual(c1->first, c2->second) && PureValuesEqual(c1->second, c2->first));
		if (!same_ops) {
			return std::nullopt;
		}
		return (c1->kind == c2->kind) ? !flip : flip;
	}
	const auto ord_group = [](CanonicalCmpKind k) {
		if (k == CanonicalCmpKind::ULt32 || k == CanonicalCmpKind::ULe32) return 1;
		if (k == CanonicalCmpKind::SLt32 || k == CanonicalCmpKind::SLe32) return 2;
		return 0;
	};
	if (ord_group(c1->kind) != 0 && ord_group(c1->kind) == ord_group(c2->kind)) {
		if (c1->kind == c2->kind) {
			if (PureValuesEqual(c1->first, c2->first) && PureValuesEqual(c1->second, c2->second)) {
				return !flip;
			}
			return std::nullopt;
		}
		// Complementary ordered comparisons have reversed operands: (x < y) vs (y <= x).
		if (PureValuesEqual(c1->first, c2->second) && PureValuesEqual(c1->second, c2->first)) {
			return flip;
		}
	}
	return std::nullopt;
}

void AppendFact(std::vector<DominatedFact>& facts, Value pred,
                std::optional<CFG::BranchCondition> vote_kind, bool truth, uint32_t depth = 0) {
	pred = pred.Resolve();
	if (depth < 8u) {
		if (const auto* inst = pred.TryInstruction()) {
			if (inst->GetOpcode() == ValueOpcode::LogicalNot) {
				AppendFact(facts, inst->Arg(0), vote_kind, !truth, depth + 1u);
				return;
			}
			if ((truth && inst->GetOpcode() == ValueOpcode::LogicalAnd) ||
			    (!truth && inst->GetOpcode() == ValueOpcode::LogicalOr)) {
				AppendFact(facts, inst->Arg(0), vote_kind, truth, depth + 1u);
				AppendFact(facts, inst->Arg(1), vote_kind, truth, depth + 1u);
				return;
			}
		}
	}
	facts.push_back({pred, vote_kind, truth});
}

std::optional<bool> EvalPredicate(const std::vector<DominatedFact>& facts, Value query_pred,
                                  std::optional<CFG::BranchCondition> query_vote_kind,
                                  uint32_t depth = 0) {
	query_pred = query_pred.Resolve();
	if (query_pred.IsImmediate() && query_pred.GetType() == Type::U1) {
		return query_pred.U1();
	}
	if (depth < 8u) {
		if (const auto* inst = query_pred.TryInstruction()) {
			if (inst->GetOpcode() == ValueOpcode::LogicalNot) {
				const auto inner =
				    EvalPredicate(facts, inst->Arg(0), query_vote_kind, depth + 1u);
				return inner.has_value() ? std::optional<bool>(!*inner) : std::nullopt;
			}
			if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
				const auto a = EvalPredicate(facts, inst->Arg(0), query_vote_kind, depth + 1u);
				if (a.has_value() && !*a) return false;
				const auto b = EvalPredicate(facts, inst->Arg(1), query_vote_kind, depth + 1u);
				if (b.has_value() && !*b) return false;
				if (a.has_value() && b.has_value()) return *a && *b;
			}
			if (inst->GetOpcode() == ValueOpcode::LogicalOr) {
				const auto a = EvalPredicate(facts, inst->Arg(0), query_vote_kind, depth + 1u);
				if (a.has_value() && *a) return true;
				const auto b = EvalPredicate(facts, inst->Arg(1), query_vote_kind, depth + 1u);
				if (b.has_value() && *b) return true;
				if (a.has_value() && b.has_value()) return *a || *b;
			}
		}
	}
	for (const auto& fact: facts) {
		const auto rel = ComparePredicates(fact.predicate, query_pred);
		if (!rel.has_value()) {
			continue;
		}
		const bool exact_scalar_vote =
		    fact.vote_kind == query_vote_kind && *rel;
		if (!exact_scalar_vote &&
		    !(IsSubgroupUniformExpr(fact.predicate) && IsSubgroupUniformExpr(query_pred))) {
			continue;
		}
		return *rel ? fact.truth : !fact.truth;
	}
	return std::nullopt;
}

} // namespace

void RemoveIdentities(Program& program) {
	// Resolve metadata before deleting its identity nodes; retain only folded native constants.
	for (auto& write: program.scalar_writes) {
		const auto value = write.value.Resolve();
		write.value = value.IsImmediate() && value.GetType() == Type::U32 ? value : Value {};
	}
	for (auto* block: program.blocks) {
		block->condition       = block->condition.Resolve();
		block->indirect_target = block->indirect_target.Resolve();
	}
	for (auto* block: program.blocks) {
		auto& instructions = block->Instructions();
		for (auto inst = instructions.begin(); inst != instructions.end();) {
			if (inst->GetOpcode() != ValueOpcode::Identity) {
				inst++;
				continue;
			}
			const auto replacement = inst->Arg(0);
			inst->ReplaceUsesWith(replacement, false);
			inst = instructions.erase(inst);
		}
	}
}

void EliminateDeadCode(const BlockList& blocks) {
	bool changed;
	do {
		std::vector<Inst*> pending;
		for (auto* block: blocks) {
			for (auto& inst: *block) {
				// Retained resource-planning instructions live outside the block list.
				inst.live = inst.MayHaveSideEffects() || std::ranges::any_of(inst.Uses(),
				    [](const Use& use) { return use.user->Parent() == nullptr; });
				if (inst.live) pending.push_back(&inst);
			}
		}
		while (!pending.empty()) {
			const auto* inst = pending.back();
			pending.pop_back();
			for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
				auto* input = inst->Arg(arg).TryInstruction();
				if (input != nullptr && input->Parent() != nullptr && !input->live) {
					input->live = true;
					pending.push_back(input);
				}
			}
		}
		// Unlink every dead operand before freeing nodes in a dead Phi cycle.
		for (auto* block: blocks)
			for (auto& inst: *block)
				if (!inst.live) inst.Invalidate();
		for (auto* block: blocks)
			std::erase_if(block->Instructions(), [](const Inst& inst) { return !inst.live; });

		changed = false;
		for (auto* block: blocks) {
			const auto& term = block->terminator;
			if (term.kind != CFG::TerminatorKind::ConditionalBranch || term.loop_header ||
			    block->condition.IsImmediate()) continue;
			for (const bool take_true: {false, true}) {
				const auto* empty = take_true ? term.false_block : term.true_block;
				const auto* merge = take_true ? term.true_block : term.false_block;
				if (empty == nullptr || merge == nullptr || !empty->empty() ||
				    empty->terminator.loop_header || merge->terminator.loop_header ||
				    empty->terminator.kind != CFG::TerminatorKind::Branch ||
				    empty->terminator.true_block != merge ||
				    std::ranges::any_of(*merge, [](const Inst& inst) {
					    return inst.GetOpcode() == ValueOpcode::Phi;
				    })) continue;
				// Keep the CFG and SSA edges intact; only the empty detour's predicate is dead.
				for (auto& inst: *block) {
					if (inst.GetOpcode() == ValueOpcode::Reference &&
					    inst.Arg(0).Resolve() == block->condition.Resolve())
						inst.SetArg(0, Value(take_true));
				}
				block->condition = Value(take_true);
				changed = true;
				break;
			}
		}
		changed |= FoldDominatedBranchConditions(blocks);
	} while (changed);
}

bool FoldDominatedBranchConditions(const BlockList& blocks) {
	const size_t n = blocks.size();
	if (n < 2u) {
		return false;
	}
	size_t dynamic_branches = 0u;
	for (const auto* block: blocks) {
		if (block != nullptr && block->terminator.kind == CFG::TerminatorKind::ConditionalBranch &&
		    !block->condition.Resolve().IsImmediate()) {
			++dynamic_branches;
		}
	}
	if (dynamic_branches < 2u) {
		return false;
	}

	std::unordered_map<const Block*, size_t> index_of;
	index_of.reserve(n);
	for (size_t i = 0; i < n; ++i) {
		index_of.emplace(blocks[i], i);
	}
	const auto is_untaken_direct_edge = [](const Block* u, const Block* v) {
		if (u == nullptr || u->terminator.kind != CFG::TerminatorKind::ConditionalBranch ||
		    u->terminator.true_block == u->terminator.false_block) {
			return false;
		}
		const auto cond = u->condition.Resolve();
		if (!cond.IsImmediate() || cond.GetType() != Type::U1) {
			return false;
		}
		const auto* taken = cond.U1() ? u->terminator.true_block : u->terminator.false_block;
		return taken != v;
	};

	const size_t          words = (n + 63u) / 64u;
	std::vector<bool>     reachable(n, false);
	std::vector<uint64_t> dom(n * words, 0u);
	reachable[0] = true;
	dom[0]       = 1u;

	std::vector<uint64_t> intersected(words);
	bool                  dom_changed = true;
	while (dom_changed) {
		dom_changed = false;
		for (size_t i = 1; i < n; ++i) {
			const auto* block = blocks[i];
			if (block == nullptr) {
				continue;
			}
			bool has_pred = false;
			for (const auto* pred: block->ImmPredecessors()) {
				const auto found = index_of.find(pred);
				if (found == index_of.end()) {
					continue;
				}
				const size_t p = found->second;
				if (!reachable[p] || is_untaken_direct_edge(pred, block)) {
					continue;
				}
				if (!has_pred) {
					std::copy_n(&dom[p * words], words, intersected.data());
					has_pred = true;
				} else {
					for (size_t w = 0; w < words; ++w) {
						intersected[w] &= dom[p * words + w];
					}
				}
			}
			if (!has_pred) {
				continue;
			}
			intersected[i / 64u] |= uint64_t {1} << (i % 64u);
			if (!reachable[i] ||
			    !std::equal(intersected.begin(), intersected.end(), &dom[i * words])) {
				reachable[i] = true;
				std::copy(intersected.begin(), intersected.end(), &dom[i * words]);
				dom_changed = true;
			}
		}
	}

	const auto dominates = [&](size_t a, size_t b) {
		return reachable[a] && reachable[b] &&
		       ((dom[b * words + (a / 64u)] >> (a % 64u)) & 1u) != 0u;
	};

	struct DominatingEdge {
		size_t                              succ_idx = 0;
		ScalarCond                          cond;
		bool                                pred_truth = false;
	};
	std::vector<DominatingEdge> dominating_edges;
	for (size_t d_idx = 0; d_idx < n; ++d_idx) {
		if (!reachable[d_idx]) {
			continue;
		}
		const auto* d    = blocks[d_idx];
		const auto& term = d->terminator;
		if (term.kind != CFG::TerminatorKind::ConditionalBranch ||
		    term.true_block == term.false_block || d->condition.Resolve().IsImmediate()) {
			continue;
		}
		const auto d_cond = UnwrapScalarCondition(d->condition);
		if (!d_cond.valid) {
			continue;
		}
		for (const bool take_true: {true, false}) {
			const auto* succ       = take_true ? term.true_block : term.false_block;
			const auto  succ_found = index_of.find(succ);
			if (succ_found == index_of.end()) {
				continue;
			}
			const size_t s_idx = succ_found->second;
			if (d_idx == s_idx || !dominates(d_idx, s_idx)) {
				continue;
			}
			bool single_entry_edge = true;
			for (const auto* pred: succ->ImmPredecessors()) {
				const auto p_found = index_of.find(pred);
				if (p_found == index_of.end()) {
					continue;
				}
				const size_t p_idx = p_found->second;
				if (!reachable[p_idx] || is_untaken_direct_edge(pred, succ)) {
					continue;
				}
				if (p_idx != d_idx && !dominates(s_idx, p_idx)) {
					single_entry_edge = false;
					break;
				}
			}
			if (single_entry_edge) {
				dominating_edges.push_back(
				    {s_idx, d_cond, take_true != d_cond.negated});
			}
		}
	}
	if (dominating_edges.empty()) {
		return false;
	}

	bool                       changed = false;
	std::vector<DominatedFact> facts;
	for (size_t b_idx = 0; b_idx < n; ++b_idx) {
		if (!reachable[b_idx]) {
			continue;
		}
		auto* b = blocks[b_idx];
		if (b->terminator.kind != CFG::TerminatorKind::ConditionalBranch ||
		    b->terminator.loop_header || b->condition.Resolve().IsImmediate()) {
			continue;
		}
		const auto b_cond = UnwrapScalarCondition(b->condition);
		if (!b_cond.valid) {
			continue;
		}
		facts.clear();
		for (const auto& edge: dominating_edges) {
			if (dominates(edge.succ_idx, b_idx)) {
				AppendFact(facts, edge.cond.predicate, edge.cond.vote_kind, edge.pred_truth);
			}
		}
		if (facts.empty()) {
			continue;
		}
		const auto pred_result = EvalPredicate(facts, b_cond.predicate, b_cond.vote_kind);
		if (!pred_result.has_value()) {
			continue;
		}
		const bool branch_taken = *pred_result != b_cond.negated;
		const auto old_cond     = b->condition.Resolve();
		for (auto& inst: *b) {
			if (inst.GetOpcode() == ValueOpcode::Reference &&
			    inst.Arg(0).Resolve() == old_cond) {
				inst.SetArg(0, Value(branch_taken));
			}
		}
		b->condition = Value(branch_taken);
		changed      = true;
	}
	return changed;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
