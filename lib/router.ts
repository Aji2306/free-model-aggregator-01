// Quality-per-cost model router and per-request budget allocator.
// Automatically balances model capability (rating_milli) against cost efficiency and allocates execution budgets.

import { type Probe, type ProbedModel, type RatingRow, type RatingsFile, ratingOfModel, modelFree, meetsRequirements, capabilitiesOf, type Requirement } from './providers.ts';

export interface RouterConfig {
  maxCostPerQuery: number;
  preferHighQuality: boolean;
  costSensitivity: number;
}

export interface ScoredModelOption {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  costEstimate: number;
  valueScore: number;
}

export interface AllocationDecision {
  selected: ScoredModelOption | null;
  fallbackChain: ScoredModelOption[];
  allocatedMaxTokens: number;
  timeoutMs: number;
  difficulty: number;
}

export function estimateDifficulty(requires: Requirement, maxTokens: number): number {
  let score = 0.2;
  if (requires.tools) score += 0.3;
  if (requires.vision) score += 0.3;
  if (requires.json) score += 0.1;
  if (maxTokens > 4096) score += 0.1;
  return Math.min(1.0, Math.max(0.0, score));
}

export function summarizeDecision(decision: AllocationDecision): string {
  const providerId = decision.selected?.probe.provider.id ?? 'none';
  const modelId = decision.selected?.model.id ?? 'none';
  return `route: provider=${providerId} model=${modelId} fallbacks=${decision.fallbackChain.length} difficulty=${decision.difficulty.toFixed(2)}`;
}

export class ModelRouter {
  private config: RouterConfig;

  constructor(config: Partial<RouterConfig> = {}) {
    this.config = {
      maxCostPerQuery: config.maxCostPerQuery ?? 1.0,
      preferHighQuality: config.preferHighQuality ?? true,
      costSensitivity: config.costSensitivity ?? 0.5,
    };
  }

  public evaluateCandidates(
    probes: readonly Probe[],
    ratings: RatingsFile | null,
    requires: Requirement
  ): ScoredModelOption[] {
    const options: ScoredModelOption[] = [];

    for (const probe of probes) {
      if (!probe.ok) continue;
      for (const model of probe.models) {
        const row = ratingOfModel(ratings, probe.provider.id, model.id);
        if (!modelFree(model, row)) continue;
        if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;

        const rating = row?.rating_milli ?? 1000;
        // Estimate cost factor based on context length and provider tier
        const ctx = row?.context_length ?? model.context_length ?? 32768;
        const costEstimate = Math.max(0.1, ctx / 131072);
        
        // Quality per cost value score
        const effectiveCost = Math.pow(costEstimate, this.config.costSensitivity);
        const valueScore = this.config.preferHighQuality
          ? rating / effectiveCost
          : 1000 / effectiveCost;

        options.push({
          probe,
          model,
          rating,
          costEstimate,
          valueScore,
        });
      }
    }

    options.sort((a, b) => b.valueScore - a.valueScore);
    return options;
  }

  public planAllocation(
    options: readonly ScoredModelOption[],
    weight: 1 | 2,
    requires: Requirement = {}
  ): AllocationDecision {
    const allocatedMaxTokens = weight === 2 ? 8192 : 4096;
    const difficulty = estimateDifficulty(requires, allocatedMaxTokens);

    if (options.length === 0) {
      return {
        selected: null,
        fallbackChain: [],
        allocatedMaxTokens,
        timeoutMs: 30000,
        difficulty,
      };
    }

    const filtered = options.filter((opt) => opt.costEstimate <= this.config.maxCostPerQuery);
    const usable = filtered.length > 0 ? filtered : [...options];

    const selected = usable[0];
    const fallbackChain = usable.slice(1);
    const timeoutMs = selected.rating > 1400 ? 45000 : 25000;

    return {
      selected,
      fallbackChain,
      allocatedMaxTokens,
      timeoutMs,
      difficulty,
    };
  }
}
