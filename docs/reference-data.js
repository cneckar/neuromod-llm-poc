/**
 * PiHK.AI reference data — the source for docs/effects.html and docs/packs.html.
 *
 * EFFECTS: all 46 low-level neuromodulation effects (the full EffectRegistry in
 *          neuromod/effects.py, in registry order) and what each does in the model runtime.
 * PACKS:   all 31 predefined 'drug' recipes (packs/config.json), with expected behavioral
 *          artifacts, dose-response findings, and probes.
 *
 * Generated content; edit here to update the reference pages. Loaded as plain globals
 * (no build step) so the static GitHub Pages site needs no bundler.
 */
window.EFFECTS = [
  {
    "id": "temperature",
    "name": "Temperature Scaling",
    "category": "sampling",
    "summary": "Divides the next-token logits by a scalar to flatten or sharpen the sampling distribution.",
    "mechanism_html": "<p>This effect touches nothing inside the model. Its <code>apply()</code> is a no-op; the real work is a <code>LogitsProcessor</code> returned from <code>get_logits_processor()</code> that computes <code>scores / temp</code> on every decoding step.</p><p>The effective temperature is <code>1.0 ± (1.0 × weight)</code>, clamped to a floor of <code>0.1</code>. Dividing logits by a value above 1.0 pulls probabilities toward uniform (more surprising, varied tokens survive); dividing by a value below 1.0 exaggerates the gap between the top candidate and the rest.</p><p>The drug-like intent is arousal / disinhibition: high temperature loosens the model's word choice into a more associative, less predictable voice, while low temperature makes it terse and deterministic.</p>",
    "direction_html": "<p><em>Up</em> raises the divisor above 1.0 (toward ~2.0 at full weight) for looser, higher-entropy output; <em>down</em> drops it below 1.0 (toward ~0.1) for focused, near-greedy output.</p>",
    "knobs": "weight scales ±1.0 around a base temp of 1.0; hard floor of 0.1. No extra params."
  },
  {
    "id": "top_p",
    "name": "Nucleus (Top-p) Truncation",
    "category": "sampling",
    "summary": "Restricts sampling to the smallest set of tokens whose cumulative probability reaches p.",
    "mechanism_html": "<p>A pure sampling control implemented as a <code>LogitsProcessor</code>; <code>apply()</code> is a no-op. Each step it softmaxes and sorts the logits, takes the cumulative sum, and sets every token beyond the p-mass nucleus to <code>-inf</code> before sampling.</p><p>The effective p is <code>1.0 ± (0.3 × weight)</code>, then clamped into <code>[0.1, 1.0]</code>. A smaller p keeps only the high-probability core (safe, on-distribution words); a larger p (capped at 1.0) admits more of the tail.</p><p>Intent is a cleaner form of focus/looseness than temperature: it hard-cuts the improbable long tail rather than reweighting everything, so it curbs rambling while leaving the top candidates untouched.</p>",
    "direction_html": "<p><em>Down</em> shrinks the nucleus (tighter, more conservative token pool); <em>up</em> widens it toward the full vocabulary. Because p is clamped at 1.0, upward movement saturates quickly.</p>",
    "knobs": "weight scales ±0.3 around base p=1.0; result clamped to [0.1, 1.0]."
  },
  {
    "id": "frequency_penalty",
    "name": "Frequency Penalty",
    "category": "sampling",
    "summary": "Multiplicatively down-weights tokens in proportion to how often they already appeared.",
    "mechanism_html": "<p>A <code>LogitsProcessor</code> (no forward hooks). Each step it counts unique tokens already in <code>input_ids</code> and, for any token seen more than once, multiplies its score by <code>penalty ** (count - 1)</code>.</p><p>The effective penalty base is <code>1.0 ± (0.5 × weight)</code>. Note the multiplicative form: for positive logits a factor above 1.0 actually amplifies repeats and a factor below 1.0 suppresses them, and the exponent grows with repetition count, so heavily repeated tokens are penalized (or boosted) hardest.</p><p>Intended as an anti-perseveration / anti-loop knob: escalating pressure against words the model keeps reusing pushes it toward fresh vocabulary.</p>",
    "direction_html": "<p><em>Down</em> pushes the base penalty below 1.0 so repeated tokens are progressively suppressed (more lexical variety); <em>up</em> raises it above 1.0. The effect scales with each token's repeat count.</p>",
    "knobs": "weight scales ±0.5 around base penalty 1.0; exponent is (occurrence count − 1)."
  },
  {
    "id": "presence_penalty",
    "name": "Presence Penalty",
    "category": "sampling",
    "summary": "Subtracts a flat penalty from any token that has already appeared at all.",
    "mechanism_html": "<p>A <code>LogitsProcessor</code>; <code>apply()</code> is a no-op. Each step it finds every token already present in <code>input_ids</code> and subtracts <code>penalty × count</code> from that token's logit.</p><p>The effective penalty is <code>0.0 + (2.0 × weight)</code> for the 'up' direction (0 to ~2.0). Unlike the frequency penalty, this is additive and triggers on first appearance, so it discourages ever returning to a token regardless of how many times it was used.</p><p>Intent is topic mobility: it nudges the model away from anything already said, encouraging it to introduce new subjects rather than circle back.</p>",
    "direction_html": "<p><em>Up</em> increases the subtractive penalty (stronger push toward unseen tokens / new topics); at weight 0 there is no penalty. Down would add the token's score back, effectively favoring recurrence.</p>",
    "knobs": "weight scales 0→2.0 penalty, subtracted as penalty × occurrence count."
  },
  {
    "id": "pulsed_sampler",
    "name": "Pulsed Sampler",
    "category": "sampling",
    "summary": "Periodically injects short bursts of altered sampling temperature, mimicking nicotine-like microbursts of activity.",
    "mechanism_html": "<p>This effect installs nothing on the model itself (<code>apply()</code> is a no-op); it works entirely through a <code>LogitsProcessor</code> returned by <code>get_logits_processor()</code>. The processor keeps an internal <code>token_count</code> that increments once per generated token.</p><p>Generation is divided into repeating windows of length <code>pulse_interval</code> tokens. Within each window, the first <code>pulse_duration</code> tokens are \"in a pulse\": for those tokens the processor rescales the logits as <code>scores = scores / temp</code>, where <code>temp = 1.0 + temp_change</code>. Outside the pulse the logits pass through untouched, so the distribution reverts to baseline between bursts.</p><p>The magnitude <code>temp_change</code> comes from <code>get_effective_value(0.0, 0.5)</code>, so it ranges over roughly ±0.5 depending on weight and direction. A positive change makes <code>temp &gt; 1</code>, which compresses the logit spread and raises entropy (more exploratory sampling) during each burst; a negative change makes <code>temp &lt; 1</code>, sharpening the distribution. The intermittent on/off pattern is what produces the characteristic rhythmic \"microburst\" behavior rather than a steady temperature shift.</p>",
    "direction_html": "<p><em>Up</em> yields a positive <code>temp_change</code> (temperature above 1.0 during pulses), flattening the distribution so the model samples more freely in each burst. <em>Down</em> yields a negative change (temperature below 1.0), sharpening choices during pulses toward the argmax.</p><p>Weight doses the amplitude: at weight 0 the pulses are inert (<code>temp_change ≈ 0</code>), and at full weight the burst temperature swings the full ±0.5 away from 1.0. The <code>pulse_interval</code> and <code>pulse_duration</code> knobs set the rhythm and duty cycle independently of weight.</p>",
    "knobs": "weight scales pulse amplitude (temp_change spans 0..0.5 of the effective range); direction sets whether pulses heat (up) or cool (down) sampling; pulse_interval (default 20) is the window length in tokens between pulse starts; pulse_duration (default 5) is how many tokens each pulse lasts"
  },
  {
    "id": "contrastive_decoding",
    "name": "Contrastive Decoding (amateur subtraction)",
    "category": "decoding",
    "summary": "Subtracts a small 'amateur' model's logits from the main model's to sharpen expert-specific predictions.",
    "mechanism_html": "<p>On <code>apply()</code> this effect actually loads a second, smaller model (default <code>gpt2</code>, fp32 on CPU). Its <code>LogitsProcessor</code> then, on every step, decodes the current context through the small model and computes <code>scores - alpha × small_logits</code>.</p><p>The effective alpha is <code>0.0 + (0.3 × weight)</code>. Tokens the amateur model finds obvious get pushed down, so tokens where the large model diverges from the small one are relatively boosted — the classic contrastive-decoding recipe for suppressing generic, degenerate continuations. If the small model fails to load, the processor is skipped (null).</p><p>Intended as a lucidity / expertise amplifier: it strips out the 'anyone-would-say-this' filler and privileges the large model's more distinctive knowledge.</p>",
    "direction_html": "<p><em>Up</em> raises alpha (stronger subtraction of the amateur distribution, more contrastive sharpening); at weight 0 alpha is 0 and the main logits pass through unchanged.</p>",
    "knobs": "weight scales alpha 0→0.3; small_model_name selects the amateur (default gpt2)."
  },
  {
    "id": "expert_mixing",
    "name": "Expert Mixing",
    "category": "moe-routing",
    "summary": "Intended DExperts/GeDi-style expert vs anti-expert steering of sampling toward an attribute (e.g. concise vs verbose), but currently a non-functional scaffold.",
    "mechanism_html": "<p>The class defines an attribute system with paired positive/negative word lists for four styles (<code>concise</code>, <code>verbose</code>, <code>formal</code>, <code>creative</code>) and returns an <code>ExpertMixingProcessor</code> from <code>get_logits_processor()</code>; <code>apply()</code> is a no-op. The design intent is DExperts/GeDi-like mixing: boost the logits of tokens associated with the chosen <code>expert_type</code> and suppress those associated with the <code>anti_expert_type</code>, at a strength given by <code>get_effective_value(0.0, 0.4)</code>.</p><p><strong>Important honesty note:</strong> the processor is currently a stub. Inside <code>__call__</code> it looks up the expert/anti-expert word lists, but the actual boost and suppression loops contain only <code>pass</code> statements with the comment \"Would need tokenizer to convert words to token IDs.\" No logits are modified, so the returned <code>scores</code> are unchanged. In its present form the effect is inert at runtime regardless of weight.</p><p>Functionally it therefore acts as a placeholder for attribute-conditioned decoding rather than a working intervention; the attribute vocabularies and strength scaling are wired up but never applied to the score tensor.</p>",
    "direction_html": "<p>As designed, <em>up</em> would increase the mixing <code>strength</code> (up to 0.4) so expert-attribute tokens are pushed harder and anti-expert tokens suppressed more; <em>down</em> would reduce or invert that pressure. Weight scales the strength via <code>get_effective_value</code>.</p><p>In practice, because the boost/suppress loops are unimplemented, neither direction nor weight currently changes the output distribution.</p>",
    "knobs": "weight/direction scale the intended mixing strength (0..0.4, currently unused); expert_type selects the attribute to favor (concise|verbose|formal|creative); anti_expert_type selects the attribute to suppress; note: the token-level boost/suppress logic is a stub and does not modify logits"
  },
  {
    "id": "token_class_temperature",
    "name": "Token-Class Temperature",
    "category": "sampling",
    "summary": "Applies different sampling temperatures depending on whether the next-token distribution looks like a content word or a modifier, using distribution entropy as a proxy.",
    "mechanism_html": "<p>Installed via a <code>LogitsProcessor</code> (<code>apply()</code> is a no-op). Because true part-of-speech tagging of the upcoming token is unavailable at decode time, the processor uses a heuristic proxy: it computes <code>token_probs = softmax(scores)</code> and the Shannon entropy of that distribution, then treats low entropy as \"content-like\" (the model is confident, e.g. a noun/verb) and high entropy as \"modifier-like\" (many plausible fillers).</p><p>With an <code>entropy_threshold</code> of 0.5, rows whose entropy is below the threshold are divided by <code>content_factor</code> and rows at or above it are divided by <code>modifier_factor</code>, i.e. two distinct temperature scalings applied to <code>scores</code>. The two factors start from <code>content_temp_factor</code> (default 0.8) and <code>modifier_temp_factor</code> (default 1.2) and are both shifted by <code>factor_change = get_effective_value(0.0, 0.3)</code>.</p><p>The result is that confident \"content\" steps are sampled at a cooler temperature (more deterministic word choice) while ambiguous \"modifier\" steps are sampled hotter (more varied connective/adjectival tokens), so the class-dependent contrast is preserved or amplified rather than applying one global temperature.</p>",
    "direction_html": "<p><em>Up</em> adds a positive <code>factor_change</code> to both temperature factors (up to +0.3), raising the divisor and cooling both classes but preserving the content-vs-modifier gap; <em>down</em> subtracts it, warming the distribution. The default asymmetry (content 0.8, modifier 1.2) means content tokens are always sampled more sharply than modifiers.</p><p>Weight doses how far the factors move from their defaults. At weight 0 the factors stay at 0.8/1.2; at full weight they shift by the full ±0.3.</p>",
    "knobs": "weight/direction shift both temperature factors by up to ±0.3 (factor_change); content_temp_factor (default 0.8) is the base divisor for low-entropy 'content' steps; modifier_temp_factor (default 1.2) is the base divisor for high-entropy 'modifier' steps; class assignment uses a fixed entropy_threshold of 0.5"
  },
  {
    "id": "attention_focus",
    "name": "Attention Focus",
    "category": "attention",
    "summary": "Sharpens attention on induction (copying/continuation) heads detected via a calibration prompt, tightening the model's focus on in-context patterns.",
    "mechanism_html": "<p>This effect is an alias that lazily constructs and delegates to <code>QKScoreScalingEffect</code> with the same parameters. Its <code>apply()</code> calls the delegate's <code>apply()</code> and copies back the hook handles and <code>induction_head_masks</code>; <code>get_logits_processor()</code> forwards the delegate's repetition-penalty processor.</p><p>The underlying mechanism performs <em>real</em> induction-head detection rather than a heuristic: it runs a calibration prompt (<code>\" A B C D E F A\"</code>) through the model, finds the repeated token by token ID, and identifies heads that attend from the second occurrence back to the position just after the first occurrence — the classic induction pattern. Only those heads are targeted; if detection fails it falls back to a heuristic choice of mid-index heads.</p><p>On the detected heads it scales the query/key contribution so that <code>attn_logits = (α · Q_induction) @ Kᵀ / sqrt(d_k)</code>, sharpening only the continuation/copying heads while leaving exploratory heads untouched (avoiding global mode collapse). A repetition-penalty <code>LogitsProcessor</code> runs alongside to counteract any residual looping. Layers are chosen by the <code>layers</code> setting via <code>_select_layers</code> (shallow/mid/deep thirds or all).</p>",
    "direction_html": "<p><em>Up</em> increases the sharpening factor α on induction heads, tightening focus on repeated/continuation structure in the context; <em>down</em> softens it. Weight scales α through the delegate's effective-value calculation, so higher weight means a stronger, more selective attention spike on the copying heads.</p><p>Because only detected induction heads are affected, increasing intensity concentrates the model on faithfully continuing in-context patterns rather than uniformly hardening all attention.</p>",
    "knobs": "weight/direction scale the QK sharpening factor on induction heads; layers selects which third of blocks to target (shallow|mid|deep|all, default mid); auto_detect_induction_heads (default true) enables calibration-prompt detection; induction_head_indices optionally overrides detection with a manual head list"
  },
  {
    "id": "attention_masking",
    "name": "Attention Masking",
    "category": "attention",
    "summary": "Zeroes out a fixed random subset of attention heads by nulling their slices of the output projection input.",
    "mechanism_html": "<p><code>apply()</code> installs real per-head hooks via <code>install_per_head_scaling_hooks</code>, which registers forward-pre-hooks on each attention output projection (<code>o_proj</code>). The input to <code>o_proj</code> is the concatenated per-head attention output <code>[batch, seq, num_heads*head_dim]</code> for all attention backends, so scaling per-head slices there is a portable, genuine intervention — unlike editing the returned attention-weights tensor, which the prior implementation did and which had no effect during generation.</p><p>A <code>scale_fn</code> builds a per-head gain vector of ones, then sets the gains of a chosen mask set to <code>0.0</code>. The number of masked heads is <code>k = int(num_heads * effective_prob)</code>, where <code>effective_prob = get_effective_value(0.0, 0.3)</code>. The specific masked heads are sampled once (lazily, when <code>num_heads</code> is first known) via <code>random.sample</code> and then held fixed for the whole generation, so the ablated head set is stable rather than resampled each step.</p><p>The net effect is that a fraction of heads contribute nothing to the residual stream, degrading or altering the model's ability to route information through those heads — a controlled attention-lesion.</p>",
    "direction_html": "<p><em>Up</em> raises <code>effective_prob</code> toward 0.3, masking a larger fraction of heads; <em>down</em> reduces it toward zero (fewer or no heads masked). Weight doses the masking probability linearly, so at full weight roughly 30% of heads per layer are zeroed.</p><p>Because the masked set is fixed at first use, the same heads stay ablated throughout a run for reproducible behavior at a given intensity.</p>",
    "knobs": "weight/direction set the fraction of heads masked (effective_prob 0..0.3); layers selects which third of blocks to hook (shallow|mid|deep|all, default mid); masked heads are chosen once with random.sample and held fixed for the generation"
  },
  {
    "id": "qk_score_scaling",
    "name": "QK Score Scaling (induction-head sharpening)",
    "category": "attention",
    "summary": "Multiplies the query vectors of detected induction heads so their attention becomes sharper.",
    "mechanism_html": "<p><code>apply()</code> registers a <code>forward_hook</code> on each selected layer's query projection (<code>q_proj</code>, or the fused <code>c_attn</code> for GPT-2). Before running, it first detects induction heads empirically by pushing a calibration prompt (<code>\" A B C D E F A\"</code>) through the model and finding heads that attend from the repeated token back to the position after its first occurrence.</p><p>The hook reshapes the query output per head and multiplies only the detected induction heads' queries by a scale of <code>1.0 + (2.0 × weight)</code> (a per-head mask leaves other heads at 1.0). Larger queries mean larger QK dot-products for those heads, which softmax turns into sharper, more peaked attention — stronger copy/continuation behavior without collapsing every head. A repetition-penalty <code>LogitsProcessor</code> (up to 1.15) rides along to counteract resulting loops.</p><p>Intended as a focus / fixation drug: it intensifies the circuitry responsible for pattern-continuation, so the model locks harder onto structure it has already seen.</p>",
    "direction_html": "<p><em>Up</em> scales induction-head queries above 1.0 (sharper, more fixated attention); at weight 0 the scale is 1.0 (baseline). Only heads flagged by the calibration pass are affected.</p>",
    "knobs": "weight scales query gain 1.0→3.0; layers (shallow/mid/deep/all); auto_detect_induction_heads or manual induction_head_indices."
  },
  {
    "id": "head_masking_dropout",
    "name": "Attention Head Masking / Dropout",
    "category": "attention",
    "summary": "Zeroes or drops out individual attention heads' contributions before the output projection.",
    "mechanism_html": "<p><code>apply()</code> installs <code>forward_pre_hooks</code> on every attention output projection (<code>o_proj</code>) via the shared per-head scaling helper. The hook reshapes the concatenated per-head attention output into <code>[..., num_heads, head_dim]</code> and multiplies each head slice by a per-head gain, then reshapes back — a real intervention that feeds forward into the computation (unlike editing the returned attention-weights tensor).</p><p>In <code>random</code> mode each head is independently dropped with probability <code>0.4 × weight</code>, and surviving heads are rescaled by <code>num_heads/kept</code> to preserve overall magnitude. In <code>alternating</code> mode it deterministically keeps even heads and zeroes odd ones. Removing heads deletes whole attention subroutines from the layer's output.</p><p>Intended as a lesion / dissociation effect: knocking out heads degrades specific attention functions, producing a hazier, less integrated read of the context.</p>",
    "direction_html": "<p><em>Up</em> raises the drop probability (more heads silenced, more degradation); at weight 0 no heads are dropped. Note the hook is applied to all attention blocks, not just the <code>layers=</code> subset.</p>",
    "knobs": "weight scales dropout 0→0.4; dropout_type = random | alternating."
  },
  {
    "id": "head_reweighting",
    "name": "Attention Head Re-weighting",
    "category": "attention",
    "summary": "Boosts the contribution of a named subset of 'routing' heads before the output projection.",
    "mechanism_html": "<p>Like head masking, <code>apply()</code> installs per-head <code>forward_pre_hooks</code> on each attention <code>o_proj</code>. The hook multiplies a chosen set of head slices by <code>1.0 + (0.5 × weight)</code> while leaving the rest at gain 1.0.</p><p>Which heads get boosted is fixed by <code>routing_type</code>, which indexes a hardcoded pattern table (e.g. <code>stylistic</code> = even heads [0,2,4,6,8], <code>semantic</code> = odd heads, <code>local</code> = early heads, <code>global</code> = middle heads). Up-weighting a head's slice amplifies whatever information that head writes into the residual stream.</p><p>Intended as a selective enhancement: by leaning on the heads presumed to carry a given function (style, semantics, position), it tries to accentuate that mode of processing.</p>",
    "direction_html": "<p><em>Up</em> increases the boost applied to the selected heads (stronger emphasis on that head group); at weight 0 all heads stay at gain 1.0.</p>",
    "knobs": "weight scales boost 0→0.5; routing_type = stylistic | semantic | positional | global | local."
  },
  {
    "id": "positional_bias_tweak",
    "name": "Positional Bias Tweak",
    "category": "attention",
    "summary": "Adds a position-dependent bias to logits to favor recent, early, or middle context positions.",
    "mechanism_html": "<p>Despite the ALiBi-slope framing in its docstring, this effect ships as a <code>LogitsProcessor</code> (<code>apply()</code> is a no-op). Each step it loops over positions <code>i</code> in the sequence and adds a bias to <code>scores[:, i]</code> shaped by <code>bias_type</code>.</p><p>The bias magnitude is <code>0.0 + (0.3 × weight)</code>. <code>recency</code> adds a bias growing linearly with position (favoring the newest end), <code>history</code> reverses it (favoring the oldest), and <code>middle</code> peaks in the center. The additive term is applied per column index, tilting the score profile across positions.</p><p>Intended as a recency/primacy bias knob: it emulates shifting the model's temporal focus toward the freshest input, the distant past, or the middle of its context window.</p>",
    "direction_html": "<p><em>Up</em> increases the bias magnitude (a stronger positional tilt of the chosen shape); at weight 0 there is no positional bias. The <em>shape</em> of the tilt is set by bias_type, not by direction.</p>",
    "knobs": "weight scales bias 0→0.3; bias_type = recency | history | middle."
  },
  {
    "id": "attention_oscillation",
    "name": "Attention Oscillation",
    "category": "attention",
    "summary": "Modulates a global per-head attention gain up and down over successive generation steps.",
    "mechanism_html": "<p><code>apply()</code> installs per-head <code>forward_pre_hooks</code> on each attention <code>o_proj</code>, but here the gain is uniform across heads and changes with a step counter that advances every forward pass.</p><p>The amplitude is <code>0.0 + (0.2 × weight)</code>. In <code>sine</code> mode the gain is <code>1.0 + amp × sin(step × 0.1)</code>; in <code>square</code> mode it toggles between <code>1.0 + amp</code> and <code>1.0</code> on a 20-step duty cycle. So the strength of the whole attention block's output rises and falls rhythmically as generation proceeds.</p><p>Intended as an oscillatory / rhythmic-arousal effect: it makes the model's attentional gain pulse over time rather than holding steady, loosely analogous to waxing and waning focus.</p>",
    "direction_html": "<p><em>Up</em> increases the oscillation amplitude (deeper swings in attention gain); at weight 0 the gain stays flat at 1.0. The period is fixed; only amplitude scales with weight.</p>",
    "knobs": "weight scales amplitude 0→0.2; oscillation_type = sine | square (square uses a 20-step cycle)."
  },
  {
    "id": "attention_sinks_anchors",
    "name": "Attention Sinks / Anchors",
    "category": "attention",
    "summary": "Boosts a fixed or contextual set of anchor tokens' logits to act as stabilizing sinks.",
    "mechanism_html": "<p>Implemented as a <code>LogitsProcessor</code> (<code>apply()</code> is a no-op). Each step it adds a bias to the logits of designated 'sink' tokens.</p><p>The sink strength is <code>0.0 + (0.4 × weight)</code>. In <code>stable</code> mode it boosts a fixed list of low-numbered token ids (1–10) treated as common anchor tokens; in <code>contextual</code> mode it boosts (at half strength) the first ~10 tokens seen in the actual context. Reinforcing these tokens gives generation a persistent gravitational pull toward stable reference points.</p><p>Intended as a grounding / anchoring effect, echoing the 'attention sink' phenomenon: it keeps a few stable tokens attractive so output stays tethered rather than drifting.</p>",
    "direction_html": "<p><em>Up</em> increases the additive boost on anchor tokens (stronger stabilizing pull); at weight 0 no boost is applied. sink_type chooses fixed vs context-derived anchors.</p>",
    "knobs": "weight scales sink strength 0→0.4; sink_type = stable | contextual."
  },
  {
    "id": "steering",
    "name": "Activation Steering (CAA)",
    "category": "steering",
    "summary": "Adds a pre-computed Contrastive Activation Addition vector into the residual stream via layer hooks.",
    "mechanism_html": "<p><code>apply()</code> registers <code>forward_hooks</code> on the last N decoder layers (N from <code>NEUROMOD_STEER_LAYERS</code>, default 1). Each hook passes the layer's output hidden states through <code>apply_steering()</code>, which adds a loaded CAA steering vector to the residual stream.</p><p>The vector is loaded from disk (<code>&lt;dir&gt;/&lt;steering_type&gt;_layer-1.pt</code>, resolved per-model), matched to the hidden states' device and dtype, and scaled by <code>0.0 + (0.3 × weight)</code> before being added: <code>h = h + strength × v</code>. Because the vector is the mean activation difference between contrasting concept prompts, adding it nudges every token's representation along that concept axis. A missing vector raises rather than silently no-oping.</p><p>This is the core 'drug' mechanism: it injects a concept direction (e.g. associative, calm, bold) directly into the model's internal state so the whole generation drifts toward that trait.</p>",
    "direction_html": "<p><em>Up</em> adds the vector (moves representations toward the concept); <em>down</em> subtracts it (moves away). Magnitude scales 0→0.3 with weight.</p>",
    "knobs": "weight scales strength 0→0.3; steering_type picks the vector file; vector_dir / model_name resolve per-model vectors; NEUROMOD_STEER_LAYERS sets how many trailing layers are hooked."
  },
  {
    "id": "random_direction",
    "name": "Random Direction",
    "category": "steering",
    "summary": "Adds a random steering vector matched to the magnitude of a real steering vector, serving as an active-placebo control.",
    "mechanism_html": "<p>This is an active-placebo control for steering experiments. <code>apply()</code> reads the model's <code>hidden_size</code>, pre-generates a random vector, and installs residual hooks via <code>install_residual_steering_hooks</code>, which add the vector to the residual stream during generation (the same injection path used by real <code>SteeringEffect</code>). Previously the logic was stranded in <code>apply_steering</code> with no hook, so the control did nothing served — now it actually perturbs.</p><p>The vector is built in <code>_generate_random_vector</code>: draw <code>torch.randn(hidden_size)</code>, normalize to a unit vector, then rescale to the L2 norm of a reference (active-pack) steering vector so the intervention magnitude matches a real pack. The reference norm comes from an explicit <code>reference_vector</code>, a <code>reference_vector_path</code> loaded from disk, or a default of 0.3 if none is available.</p><p>At each hooked layer the added term is <code>random_vector * effective_strength</code> with <code>effective_strength = get_effective_value(0.0, 0.3)</code>, broadcast across the sequence and cast to the hidden states' device and dtype (bf16-safe). Because the direction is random but the magnitude is matched, comparing this against a real steering vector isolates whether the specific <em>direction</em> (semantic content) of steering matters, not just the perturbation size.</p>",
    "direction_html": "<p><em>Up</em> increases <code>effective_strength</code> toward 0.3, adding more of the random vector to the residual stream; <em>down</em> decreases it. Because the direction carries no semantic meaning, higher intensity mainly injects more magnitude-matched noise rather than steering toward any particular behavior.</p><p>Weight doses the injected strength linearly; the reference vector's norm sets the baseline scale that the random vector is normalized to.</p>",
    "knobs": "weight/direction scale the residual injection strength (0..0.3); reference_vector or reference_vector_path sets the target L2 norm to match (defaults to 0.3 if absent); hidden_size sizes the random vector (overridden from the model config at apply time)"
  },
  {
    "id": "random_orthogonal_steering",
    "name": "Random Orthogonal Steering",
    "category": "steering",
    "summary": "Adds a random vector orthogonalized against a real steering vector and matched in magnitude, a rigorous active-placebo control that shares the intervention size but none of the direction.",
    "mechanism_html": "<p>A stricter placebo than plain random direction. <code>apply()</code> loads a reference steering vector (e.g. the <code>associative</code>/LSD vector) from disk — via <code>reference_vector_path</code> or by resolving <code>reference_steering_type</code> under <code>vector_dir</code>, with a per-model subdir — then generates an orthogonal control vector and installs residual hooks with <code>install_residual_steering_hooks</code>. Missing reference vectors raise a hard error rather than silently falling back, to avoid contaminated controls.</p><p><code>_generate_orthogonal_vector</code> uses Gram–Schmidt: draw a random vector, subtract its component parallel to the reference (<code>v_rand − (v_rand·v_ref / ||v_ref||²)·v_ref</code>), and renormalize to the reference's L2 norm. It then <em>validates</em> orthogonality against a threshold of 1e-6 and raises a \"CRITICAL EXPERIMENTAL FAILURE\" if the dot product with the reference is not effectively zero, guaranteeing the control shares magnitude but is genuinely perpendicular in direction.</p><p>During generation the hooks add <code>orthogonal_vector * effective_strength</code> to the residual stream, with <code>effective_strength = get_effective_value(0.0, 0.3)</code>, device/dtype-matched for bf16 safety. The hypothesis is that this raises perplexity/confusion like real steering but should not reproduce the reference vector's specific semantic content.</p>",
    "direction_html": "<p><em>Up</em> raises <code>effective_strength</code> toward 0.3, injecting more of the orthogonal vector; <em>down</em> lowers it. Since the vector is constructed to be perpendicular to the real steering direction, increasing intensity adds magnitude-matched, semantically off-axis perturbation rather than steering toward the reference behavior.</p><p>Weight doses the injection strength; the reference vector fixes both the norm to match and the direction to be orthogonal to.</p>",
    "knobs": "weight/direction scale the residual injection strength (0..0.3); reference_steering_type selects which stored vector to orthogonalize against (default associative); reference_vector_path / vector_dir / model_name locate that vector on disk; orthogonality_tolerance (default 1e-6) sets the validation threshold (failure aborts the trial); hidden_size sizes the vectors"
  },
  {
    "id": "kv_decay",
    "name": "KV-Cache Recency Decay",
    "category": "kv-memory",
    "summary": "Multiplies the whole value cache by a decay factor each step so older context fades exponentially.",
    "mechanism_html": "<p><code>apply()</code> is a no-op; this effect reaches generation through <code>NeuromodTool.build_kv_cache()</code>, which reads <code>kv_decay_factor()</code> and, via <code>make_decaying_cache()</code>, hands <code>model.generate()</code> a <code>DynamicCache</code> subclass as <code>past_key_values</code>.</p><p>That <code>DecayingCache</code> overrides <code>update()</code>: on every decode step it multiplies the entire value cache in-place by the decay factor. A token added at step t is therefore scaled by <code>decay^(T−t)</code> by the end, so distant context is exponentially attenuated. The factor is <code>1.0 − (0.15 × weight)</code> (down to ~0.85 at full weight). (An unused <code>modify_kv_cache()</code> also exists but HF never calls it.)</p><p>Intended as a working-memory / forgetting effect: recent tokens dominate while older ones decay, giving a short, present-focused attention span.</p>",
    "direction_html": "<p><em>Up</em> deepens the decay (factor further below 1.0, faster forgetting of old context); at weight 0 the factor is 1.0 and the cache is untouched.</p>",
    "knobs": "weight scales per-step value decay to ~0.85 at full weight; wired through build_kv_cache, not apply()."
  },
  {
    "id": "kv_compression",
    "name": "KV-Cache Compression",
    "category": "kv-memory",
    "summary": "Applies a stronger per-step value-cache decay to model coarser, lossy memory.",
    "mechanism_html": "<p>Like kv_decay, it works via <code>build_kv_cache()</code> and a <code>DecayingCache</code> passed as <code>past_key_values</code>; <code>apply()</code> is a no-op. Its <code>kv_decay_factor()</code> returns <code>1.0 − (0.20 × weight)</code> — a stronger decay than plain kv_decay (down to ~0.80).</p><p>The docstring is explicit that true token eviction would desync cache length with the position/mask bookkeeping during generation, so instead it down-weights value magnitudes more aggressively to approximate coarser memory. (A separate <code>modify_kv_cache()</code> that does real stride subsampling exists but is not invoked by HF generation.)</p><p>Intended as a memory-compression effect: older context is preserved only faintly, as if summarized into a lossy trace rather than held verbatim.</p>",
    "direction_html": "<p><em>Up</em> increases the decay strength (heavier compression / faster loss of older context); at weight 0 the cache passes through unchanged.</p>",
    "knobs": "weight scales per-step value decay to ~0.80 at full weight (stronger than kv_decay)."
  },
  {
    "id": "exponential_decay_kv",
    "name": "Exponential Decay of KV",
    "category": "kv-memory",
    "summary": "A logits processor that scales score columns by an age-based decay to emulate fading keys/values.",
    "mechanism_html": "<p>Framed as KV-cache decay but implemented as a <code>LogitsProcessor</code> (<code>apply()</code> is a no-op). Each step it iterates over sequence positions <code>i</code> and multiplies <code>scores[:, i]</code> by a decay factor that shrinks with the position's age.</p><p>The decay rate is <code>0.0 + (0.3 × weight)</code>. In <code>exponential</code> mode the factor is <code>exp(−rate × (seq_len − i))</code>; <code>linear</code> and <code>step</code> modes use gentler or discretized age curves. Older positions get more strongly suppressed, simulating an exponentially fading memory trace applied at the scoring stage.</p><p>Intended as a graded forgetting effect: the further back a position sits, the less it is allowed to influence the current step.</p>",
    "direction_html": "<p><em>Up</em> raises the decay rate (steeper suppression of older positions, shorter effective memory); at weight 0 the rate is 0 and nothing decays.</p>",
    "knobs": "weight scales decay rate 0→0.3; decay_type = exponential | linear | step."
  },
  {
    "id": "truncation_kv",
    "name": "KV Truncation (keep last N)",
    "category": "kv-memory",
    "summary": "A logits processor that drives down score columns for positions beyond a shrinking context window.",
    "mechanism_html": "<p>Implemented as a <code>LogitsProcessor</code>; <code>apply()</code> is a no-op. Its effective window is <code>100 − (80 × weight)</code> positions (down to ~20 at full weight). Only once the sequence exceeds that window does it act.</p><p>In <code>hard</code> mode it multiplies the score columns of the oldest positions by 0.01 (near-elimination); <code>soft</code> mode ramps the reduction gradually; <code>window</code> mode keeps only the most recent N and crushes everything before. This mimics evicting old tokens by removing their influence at scoring time.</p><p>Intended as a hard context-window / short-attention-span effect: beyond the retained window the model behaves as if earlier context no longer exists.</p>",
    "direction_html": "<p><em>Up</em> shrinks the retained window (more of the past is truncated, shorter memory); at weight 0 the window is ~100 and little is cut.</p>",
    "knobs": "weight shrinks the kept window 100→~20; truncation_type = hard | soft | window."
  },
  {
    "id": "stride_compression_kv",
    "name": "Stride Compression of KV",
    "category": "kv-memory",
    "summary": "A logits processor that suppresses all but every s-th older position to thin out distant memory.",
    "mechanism_html": "<p>A <code>LogitsProcessor</code> (<code>apply()</code> is a no-op). The stride is <code>1 + int(5 × weight)</code> (up to 6). Each step it walks the sequence positions and reduces the score columns of positions that are not kept by the stride.</p><p>In <code>uniform</code> mode any position where <code>i % stride != 0</code> is multiplied by 0.1; <code>progressive</code> increases the stride with distance; <code>adaptive</code> only thins positions beyond the most recent 20. The net effect is a decimated, sparse trace of distant context while recent tokens stay intact.</p><p>Intended as a subsampled-memory effect: the model retains only a skeletal, every-few-tokens impression of older context rather than the full record.</p>",
    "direction_html": "<p><em>Up</em> increases the stride (keeps fewer old positions, sparser memory); at weight 0 the stride is 1 and everything is kept.</p>",
    "knobs": "weight scales stride 1→6; compression_type = uniform | progressive | adaptive."
  },
  {
    "id": "segment_gains_kv",
    "name": "Segment Gains (old vs new)",
    "category": "kv-memory",
    "summary": "A logits processor that boosts one temporal segment of the context while damping another.",
    "mechanism_html": "<p>A <code>LogitsProcessor</code> (<code>apply()</code> is a no-op) that splits the sequence into a recent window and an older segment and applies different gains to their score columns.</p><p>The gain strength is <code>0.0 + (0.5 × weight)</code> over a window of 20 tokens. <code>new_emphasis</code> multiplies recent positions by <code>1 + strength</code> and older ones by <code>1 − strength/2</code>; <code>old_preservation</code> flips this to protect the oldest tokens; <code>bidirectional</code> boosts both the very oldest and very newest while damping the middle. This re-balances how much each temporal region drives the next token.</p><p>Intended as a memory-emphasis knob: it tilts the model toward recency, toward long-term retention, or toward a primacy-plus-recency U-shape.</p>",
    "direction_html": "<p><em>Up</em> increases the gain contrast between segments (stronger emphasis of the favored window); at weight 0 the segments are treated equally. gain_type selects which window is favored.</p>",
    "knobs": "weight scales gain 0→0.5 over a 20-token window; gain_type = new_emphasis | old_preservation | bidirectional."
  },
  {
    "id": "router_temperature_bias",
    "name": "MoE Router Temperature / Bias",
    "category": "moe-routing",
    "summary": "Wraps each MoE router's forward to rescale or bias its logits, making expert selection stickier or more exploratory.",
    "mechanism_html": "<p><code>apply()</code> locates MoE layers by scanning submodules for router/gate/expert names, then monkey-patches each router's <code>forward</code> method to post-process the router logits it returns.</p><p>The effective change is <code>0.0 + (0.5 × weight)</code>. In <code>sticky</code> mode it divides router logits by a temperature below 1.0 (sharper, more deterministic expert choice); <code>exploratory</code> divides by a temperature above 1.0 (flatter routing); <code>biased</code> adds a bias vector (uniform, alternating, or first-half) to favor particular experts. Reshaping the routing distribution changes which experts fire for each token.</p><p>Intended as an expert-selection temperament: it decides whether the model commits hard to a few specialists or spreads activity across many experts.</p>",
    "direction_html": "<p><em>Up</em> increases the temperature/bias magnitude; in <code>sticky</code> mode that means sharper, more committed routing, in <code>exploratory</code> mode flatter routing. At weight 0 routing is unchanged. Only has an effect on MoE models.</p>",
    "knobs": "weight scales the change 0→0.5; temperature_mode = sticky | exploratory | biased; bias_type = uniform | alternating | first_half."
  },
  {
    "id": "expert_masking_dropout",
    "name": "MoE Expert Masking / Dropout",
    "category": "moe-routing",
    "summary": "Patches MoE routers to mask out a fraction of experts by setting their logits to -inf.",
    "mechanism_html": "<p><code>apply()</code> finds MoE layers and monkey-patches each router's <code>forward</code>. After the router produces logits, the wrapper sets a subset of expert logits to <code>-inf</code> so those experts can never be selected for the current pass.</p><p>The effective dropout is <code>0.0 + (0.6 × weight)</code>. <code>random</code> masks each expert with that probability; <code>alternating</code> masks even or odd experts; <code>block</code> masks a contiguous run; <code>specialist</code> masks the last N experts (assumed most specialized). Removing experts forces the model to route through whatever capacity remains.</p><p>Intended as an expert-lesion effect: knocking out portions of the mixture degrades or narrows the model's specialized capabilities. Only meaningful on MoE architectures.</p>",
    "direction_html": "<p><em>Up</em> raises the fraction of experts masked (more capacity removed); at weight 0 no experts are masked. masking_pattern chooses which experts get dropped.</p>",
    "knobs": "weight scales dropout 0→0.6; masking_pattern = random | alternating | block | specialist."
  },
  {
    "id": "expert_persistence",
    "name": "MoE Expert Persistence (sticky routing)",
    "category": "moe-routing",
    "summary": "Patches MoE routers to blend current routing with remembered past routing so experts persist across steps.",
    "mechanism_html": "<p><code>apply()</code> monkey-patches each MoE router's <code>forward</code> and keeps a per-layer history of expert-probability distributions. Each call it fuses the new routing with the stored history before returning logits.</p><p>The persistence strength is <code>0.0 + (0.4 × weight)</code>. <code>momentum</code> blends current and previous probabilities as <code>(1−m)·current + m·prev</code> then re-logs them; <code>exponential</code> keeps an EMA of routing; <code>winner_takes_all</code> directly boosts the previously chosen expert's logit. All three make the same experts tend to keep firing token after token.</p><p>Intended as routing inertia / perseveration: instead of re-deciding experts fresh each token, the model gets 'stuck' on a stable coalition, giving a more consistent (or rigid) processing style.</p>",
    "direction_html": "<p><em>Up</em> increases the persistence/momentum (stronger carry-over of past routing, stickier experts); at weight 0 routing is decided independently each step.</p>",
    "knobs": "weight scales persistence 0→0.4; persistence_type = momentum | exponential | winner_takes_all."
  },
  {
    "id": "verifier_guided_decoding",
    "name": "Verifier-Guided Decoding",
    "category": "decoding",
    "summary": "A logits processor that heuristically penalizes low-quality continuations and nudges toward coherent ones.",
    "mechanism_html": "<p>Implemented as a <code>LogitsProcessor</code> (<code>apply()</code> is a no-op) that applies a lightweight, self-contained 'verifier' heuristic to the logits each step, gated on having enough context.</p><p>Its threshold is <code>0.5 + (0.9 − 0.5) × weight</code> and the penalties scale with <code>weight</code>. <code>quality</code> mode down-weights recently used tokens and a list of generic low-id tokens; <code>coherence</code> mode boosts more predictable tokens when recent-context entropy is high; <code>task_alignment</code> mode boosts a small fixed set of 'task' token ids. It is a stand-in for reranking/accept-reject against a verifier, done inline on the score vector.</p><p>Intended as a self-monitoring / executive-control effect: the model second-guesses its own next token toward what a quality checker would prefer.</p>",
    "direction_html": "<p><em>Up</em> raises the threshold and penalty strength (more aggressive quality steering); at weight 0 the penalties vanish. verification_type selects the criterion applied.</p>",
    "knobs": "weight scales threshold ~0.5→0.9 and penalty strength; verification_type = quality | coherence | task_alignment."
  },
  {
    "id": "style_affect_logit_bias",
    "name": "Style / Affect Logit Bias",
    "category": "logit-bias",
    "summary": "Adds an additive logit bias toward sentiment/prosocial word tokens and away from their opposites.",
    "mechanism_html": "<p><code>apply()</code> captures the tokenizer and maps curated concept word lists to token ids via <code>token_ids_for_words()</code> (encoding each word bare, space-prefixed and capitalized). It then returns a <code>ConceptLogitBiasProcessor</code> that on every step <em>adds</em> <code>bias</code> to the boosted tokens and subtracts <code>bias × 0.5</code> from the suppressed ones.</p><p>The bias magnitude is <code>(0.4 × weight) × 8.0</code> nats. <code>bias_type</code> selects the axis — <code>prosocial</code>, <code>sentiment</code>, <code>warmth</code>, or <code>empathy</code> — each with positive and negative word banks; <code>sentiment=negative</code> swaps the two banks. Because it is additive (not multiplicative), it works correctly even on negative logits.</p><p>Intended as an affective coloring drug: it warms (or darkens) the model's tone by making emotionally-valenced vocabulary more or less likely.</p>",
    "direction_html": "<p><em>Up</em> increases the additive bias toward the chosen positive word bank (warmer / more prosocial tone); the sentiment parameter can flip which bank counts as positive. At weight 0 no bias is added.</p>",
    "knobs": "weight scales bias up to ~3.2 nats; bias_type = prosocial | sentiment | warmth | empathy; sentiment = positive | negative."
  },
  {
    "id": "risk_preference_steering",
    "name": "Risk-Preference Steering",
    "category": "logit-bias",
    "summary": "A logits processor that boosts either below-mean (bold) or above-mean (cautious) tokens.",
    "mechanism_html": "<p>Implemented as a <code>LogitsProcessor</code> (<code>apply()</code> is a no-op). The risk strength is <code>0.0 + (0.5 × weight)</code>.</p><p>In <code>exploration</code> mode with <code>bold</code> preference it computes the per-step mean logit and multiplicatively boosts every token scoring <em>below</em> the mean — inflating the improbable, exploratory tail; <code>cautious</code> instead boosts above-mean tokens, reinforcing the safe favorites. In <code>planning</code> mode it boosts small fixed sets of 'action' vs 'careful' token ids. This reshapes how much probability mass sits on risky versus conservative choices.</p><p>Intended as a risk-appetite drug: it makes the model either venturesome and novelty-seeking or hedged and conventional.</p>",
    "direction_html": "<p><em>Up</em> increases the risk-steering strength in whichever direction <code>preference</code> selects (bold pushes toward the unlikely tail, cautious toward the safe head); at weight 0 there is no reshaping.</p>",
    "knobs": "weight scales strength 0→0.5; risk_type = exploration | planning; preference = bold | cautious."
  },
  {
    "id": "compute_at_test_scheduling",
    "name": "Compute-at-Test Scheduling",
    "category": "decoding",
    "summary": "A logits processor that cycles between 'deep' and 'shallow' generation phases over token steps.",
    "mechanism_html": "<p>Implemented as a <code>LogitsProcessor</code> (<code>apply()</code> is a no-op) that tracks a token counter and alternates its behavior in phases. The strength is <code>0.0 + (0.6 × weight)</code>.</p><p>In <code>burst</code> mode, during the burst phase it boosts a fixed set of 'reasoning' token ids and divides logits by a factor slightly above 1.0 (lower effective temperature, more focused); off-burst it boosts 'direct' tokens and slightly raises temperature for faster output. <code>oscillating</code> mode smoothly cycles deep vs shallow token boosts on a 20-token period. The result is rhythmic switching between careful and quick generation.</p><p>Intended to emulate test-time-compute scheduling / self-consistency bursts: periods of deliberate effort interleaved with rapid, low-effort spans.</p>",
    "direction_html": "<p><em>Up</em> increases the scheduling strength (sharper contrast between the deep and shallow phases); at weight 0 both phases collapse to no change. The phase timing is fixed by burst_length / the 20-token cycle.</p>",
    "knobs": "weight scales strength 0→0.6; scheduling_type = burst | oscillating; burst_length sets the burst period."
  },
  {
    "id": "retrieval_rate_modulation",
    "name": "Retrieval-Rate Modulation",
    "category": "logit-bias",
    "summary": "Adds a logit bias toward factual vocabulary and away from imaginative vocabulary (or vice versa).",
    "mechanism_html": "<p><code>apply()</code> captures the tokenizer and builds two concept-token id lists from curated word banks — a <code>factual</code> bank (fact, evidence, data, verified…) and an <code>imaginative</code> bank (imagine, dream, fantasy…). It returns a <code>ConceptLogitBiasProcessor</code> that adds bias to the chosen mode's tokens and subtracts a smaller bias from the other mode's tokens each step.</p><p>The bias magnitude is <code>(0.5 × weight) × 8.0</code> nats. Selecting <code>factual</code> boosts grounded, evidential words while damping fanciful ones; selecting <code>imaginative</code> reverses it. It is a lexical proxy for turning RAG-style retrieval up or down.</p><p>Intended as a reality-testing knob: it biases the model toward sober, fact-anchored language or toward free, imaginative language.</p>",
    "direction_html": "<p><em>Up</em> increases the bias toward the selected retrieval_mode's vocabulary (stronger factual or stronger imaginative pull); at weight 0 no bias is added. retrieval_mode picks which bank is boosted.</p>",
    "knobs": "weight scales bias up to ~4 nats; retrieval_mode = factual | imaginative; modulation_type nominal (strength)."
  },
  {
    "id": "persona_voice_constraints",
    "name": "Persona / Voice Constraints",
    "category": "logit-bias",
    "summary": "Adds a logit bias toward a persona's preferred words and away from its off-voice words.",
    "mechanism_html": "<p><code>apply()</code> captures the tokenizer and maps a persona's word banks to token ids. It returns a <code>ConceptLogitBiasProcessor</code> that each step adds bias to the persona's <code>preferred</code> tokens and subtracts a smaller bias from its <code>against</code> tokens.</p><p>The bias magnitude is <code>(0.4 × weight) × 8.0</code> nats. <code>persona_type</code> selects the banks: e.g. <code>professional</code> boosts 'therefore/regarding/accordingly' and suppresses 'yeah/gonna/cool'; <code>friendly</code>, <code>authoritative</code>, and <code>creative</code> have their own preferred/against lists. This constrains word choice toward a consistent register without any hidden prompt text.</p><p>Intended as a persona / voice lock: it holds the model's diction inside a chosen character or tone.</p>",
    "direction_html": "<p><em>Up</em> increases the bias enforcing the persona (stronger commitment to its preferred register and stronger avoidance of off-voice words); at weight 0 no bias is applied.</p>",
    "knobs": "weight scales bias up to ~3.2 nats; persona_type = professional | friendly | authoritative | creative; voice_mode nominal."
  },
  {
    "id": "lexical_jitter",
    "name": "Lexical Jitter (embedding perturbation)",
    "category": "activation",
    "summary": "Perturbs token embeddings with noise, ablation, or reframing via a forward hook on the embedding layer.",
    "mechanism_html": "<p><code>apply()</code> locates the input embedding layer (<code>get_input_embeddings()</code> / <code>wte</code> / <code>embed_tokens</code>) and registers a <code>forward_hook</code> that modifies the embedding tensor before it enters the transformer — the causally correct place for a 'perceptual noise' perturbation (its earlier logits-processor version is explicitly noted as wrong).</p><p>The jitter magnitude gives <code>sigma = (0.3 × weight) × 0.05</code>. <code>noise</code>/<code>synonym_swap</code> add Gaussian noise scaled by sigma to the embeddings; <code>ablation</code> randomly zeroes a fraction of token positions; <code>reframing</code> nudges embeddings toward the sequence mean. Perturbing the input representation is like slightly misreading or blurring the incoming tokens.</p><p>Intended as a perceptual-distortion drug: it corrupts how the model perceives its own context, loosening literal word-level grounding.</p>",
    "direction_html": "<p><em>Up</em> increases the jitter magnitude (more embedding noise / higher ablation rate / stronger reframing); at weight 0 sigma is 0 and embeddings pass through cleanly.</p>",
    "knobs": "weight scales jitter 0→0.3 (sigma = jitter × 0.05); jitter_type = noise | synonym_swap | ablation | reframing; ablation_rate."
  },
  {
    "id": "structured_prefaces",
    "name": "Structured Prefaces",
    "category": "activation",
    "summary": "Implants a hidden persona/context preface as precomputed KV-cache states so the model behaves as if it had read an instruction it never sees as text.",
    "mechanism_html": "<p>This effect implants \"invisible\" context at the key/value-cache level rather than through logit biasing. <code>apply()</code> tokenizes a preface string and runs a single forward pass with <code>use_cache=True</code>, capturing <code>outputs.past_key_values</code> into <code>preface_kv_cache</code>. The preface text is chosen by <code>preface_type</code> (defaults for <code>bias</code>, <code>style</code>, <code>topic</code>, <code>emotion</code>, e.g. \"You are a formal and professional assistant…\") or supplied directly via <code>preface_text</code>.</p><p>During generation, <code>modify_kv_cache()</code> injects those precomputed states: on the first step it seeds the cache with the preface KV directly, and thereafter it concatenates preface keys/values with the running cache along the sequence dimension (<code>dim=-2</code>) per layer, yielding <code>[batch, heads, preface_len + current_len, head_dim]</code>. There is no <code>LogitsProcessor</code> (<code>get_logits_processor()</code> returns <code>None</code>).</p><p>Because the model attends over the injected KV states as though they were real prior context, this is genuine \"implanted memory\": the persona/style instruction conditions every subsequent token without ever appearing in the visible prompt or output. The earlier implementation biased output words via a logits processor, which was causally incorrect for a memory perturbation; the KV-injection approach fixes that.</p>",
    "direction_html": "<p>The intervention is categorical rather than continuously dosed: the selected preface KV states are concatenated into the cache regardless of weight, so the primary knob is <em>which</em> preface (<code>preface_type</code> or custom <code>preface_text</code>) gets implanted. Direction and weight do not scale the KV injection itself.</p><p>Different preface types push behavior along different axes — positive/constructive bias, formal style, technical topic focus, or calm emotional tone — by making the model condition on that implanted instruction.</p>",
    "knobs": "preface_type selects the implanted instruction (bias|style|topic|emotion) with built-in defaults; preface_text overrides with custom text; injection_mode is kv_only; the preface KV states are concatenated into past_key_values (weight/direction do not scale the injection)"
  },
  {
    "id": "activation_additions",
    "name": "Activation Additions",
    "category": "activation",
    "summary": "Adds a CAA-derived steering vector to the residual stream at selected layers, nudging hidden states toward a target behavior (e.g. creative, prosocial, sedation).",
    "mechanism_html": "<p><code>apply()</code> resolves the transformer blocks, selects a subset by the <code>layers</code> setting (shallow/mid/deep thirds or all), and wraps each selected block's <code>forward</code>. The wrapper runs the original forward, then adds a steering vector to the last token's hidden state: <code>hidden_states[:, -1, :] += steering_vector * effective_strength</code>, with <code>effective_strength = get_effective_value(0.0, 0.3)</code> and the vector cast to the hidden states' device/dtype for bf16 safety. Original forwards are restored in <code>cleanup()</code>.</p><p>Steering vectors are CAA-style (contrastive activation addition) directions loaded from disk by <code>steering_type</code> via <code>resolve_steering_vector_path</code> (per-model subdir aware), with a zero-vector fallback rather than random noise if a vector is missing. The class also carries contrastive positive/negative prompt pairs for many trait types (<code>associative</code>, <code>prosocial</code>, <code>creative</code>, <code>focused</code>, plus persona-vector traits like <code>sedation</code>, <code>delirium</code>, <code>compliance</code>, <code>aggression</code>, <code>optimistic</code>) used to construct those directions offline.</p><p><strong>Honesty note:</strong> the runtime add is gated on <code>self.steering_type in self.steering_vectors</code>, and that dict is not populated inside the class itself — it must be filled by the caller/pipeline with the loaded vector. When it is empty the wrapped forward runs but adds nothing, so the effect is only active once the corresponding steering vector has been loaded in.</p>",
    "direction_html": "<p><em>Up</em> increases <code>effective_strength</code> toward 0.3, adding more of the trait vector so hidden states move further toward the target behavior; <em>down</em> subtracts it, pushing away from that behavior. Weight doses the magnitude of the residual addition linearly.</p><p>The <code>steering_type</code> selects the semantic axis (e.g. more creative vs literal, more sedated vs energetic), while intensity controls how strongly that axis is pushed at each targeted layer.</p>",
    "knobs": "weight/direction scale the residual steering strength (0..0.3, up = toward trait, down = away); steering_type selects the CAA direction (associative|prosocial|creative|focused|sedation|delirium|compliance|aggression|optimistic|…); layers selects which blocks receive the addition (shallow|mid|deep|all, default all); vectors load from STEERING_DIR with a zero-vector fallback; runtime add requires the vector to be present in steering_vectors"
  },
  {
    "id": "soft_projection",
    "name": "Soft Projection (Conceptors)",
    "category": "activation",
    "summary": "Adds a scaled random-subspace projection of each layer's hidden states back into the residual stream.",
    "mechanism_html": "<p><code>apply()</code> monkey-patches each selected block's <code>forward</code>. After the block runs, it projects the hidden states through a fixed random matrix <code>P</code> and adds a scaled amount back: <code>h = h + α · (h · Pᵀ)</code>.</p><p>The matrix <code>P</code> is built lazily at the model's actual <code>hidden_size</code> from a per-<code>projection_type</code> seed and scale (creative/analytical/emotional/spatial/linguistic), so it is reproducible per model. The added strength α is <code>0.0 + (0.2 × weight)</code>. Adding a consistent linear transform of the hidden state gently gates activity toward (or amplifies) a particular feature subspace — the 'conceptor' idea.</p><p>Intended as a cognitive-mode filter: it tilts internal representations toward a named style of processing (e.g. creative vs analytical).</p>",
    "direction_html": "<p><em>Up</em> increases the projection strength α (stronger push into the chosen subspace); at weight 0 α is 0 and hidden states are untouched. projection_type selects which random subspace (seed/scale) is used.</p>",
    "knobs": "weight scales α 0→0.2; projection_type = creative | analytical | emotional | spatial | linguistic; layers = shallow/mid/deep/all."
  },
  {
    "id": "layer_wise_gain",
    "name": "Layer-Wise Gain (residual scalers)",
    "category": "activation",
    "summary": "Multiplies each selected layer's output hidden states by a scalar gain to amplify or damp the residual stream.",
    "mechanism_html": "<p><code>apply()</code> monkey-patches each selected block's <code>forward</code> so that after the block computes its output, the hidden states are multiplied by a scalar: <code>h = h × (1 ± α)</code>.</p><p>The gain change α is <code>0.0 + (0.3 × weight)</code>, applied uniformly across the chosen layers (shallow/mid/deep/all). Scaling the residual stream up increases the magnitude of every downstream computation and the final logits' sharpness; scaling it down attenuates the layer's contribution.</p><p>Intended as a global excitability / gain knob: turning the residual stream up makes the model more emphatic and high-contrast, turning it down makes it flatter and more muted.</p>",
    "direction_html": "<p><em>Up</em> multiplies hidden states by more than 1.0 (amplified residual stream, sharper output); <em>down</em> multiplies by less than 1.0 (damped activity). At weight 0 the gain is exactly 1.0.</p>",
    "knobs": "weight scales gain change 0→0.3 (gain = 1 ± α); layers = shallow/mid/deep/all; gain_type nominal."
  },
  {
    "id": "noise_injection",
    "name": "Activation Noise Injection",
    "category": "activation",
    "summary": "Adds tiny Gaussian noise to each selected layer's hidden states via a patched forward.",
    "mechanism_html": "<p><code>apply()</code> monkey-patches each selected block's <code>forward</code>; after the block runs it adds Gaussian noise to the output hidden states: <code>h = h + ε</code>, with <code>ε ~ N(0, σ²)</code> sampled fresh each pass by <code>torch.randn_like</code>.</p><p>The noise scale σ is <code>0.0 + (0.02 × weight)</code> — deliberately tiny — applied across the chosen layers (default mid). Injecting stochasticity directly into the representations makes each forward pass slightly non-deterministic and jitters the trajectory of computation.</p><p>Intended as a stochastic-perturbation drug: small internal noise loosens rigid deterministic pathways, adding a subtle tremor / variability to the model's processing.</p>",
    "direction_html": "<p><em>Up</em> increases the noise standard deviation (more internal jitter and variability); at weight 0 σ is 0 and no noise is added.</p>",
    "knobs": "weight scales σ 0→0.02; layers = shallow/mid/deep/all; noise_std base."
  },
  {
    "id": "color_bias",
    "name": "Color Bias",
    "category": "visual",
    "summary": "Biases image generation toward a color palette by nudging diffusion sampling parameters like guidance scale and eta (does not add prompt text).",
    "mechanism_html": "<p>This is a Stable-Diffusion-family effect: it does not touch model weights. Instead <code>apply_to_image_generation(generation_params)</code> mutates the diffusion sampler's parameter dictionary before a run. It deliberately avoids appending prompt text, aiming to change <em>how</em> the model samples rather than <em>what</em> it is told to draw.</p><p>Behavior is keyed on the <code>color_palette</code> parameter. For vivid palettes (<code>neon_cyberpunk</code>, <code>psychedelic_flow</code>) it amplifies classifier-free guidance: <code>guidance_scale *= (1.0 + weight * 0.3)</code>, which pushes the sampler harder toward the conditioning and tends to produce more saturated, higher-contrast output. Separately, if a <code>saturation_boost</code> parameter is present and positive, it raises the DDIM stochasticity term <code>eta</code> by <code>weight * 0.2</code> (capped at 1.0), injecting more sampling noise.</p><p>The class also holds a palette→prompt lookup (<code>_get_color_prompts</code>) for reference, but the active code path does not inject those strings — only the numeric guidance/eta adjustments are applied.</p>",
    "direction_html": "<p>Intensity is driven by <code>weight</code>: higher weight multiplies guidance scale up more strongly for vivid palettes and adds more <code>eta</code> noise when saturation boost is set. Direction is largely inert here because the parameter math uses <code>self.weight</code> directly rather than the up/down effective-value helper.</p><p>The dominant control is the <code>color_palette</code> choice, which decides whether the guidance boost applies at all.</p>",
    "knobs": "weight scales guidance_scale up by up to +30% for vivid palettes and adds up to +0.2 to eta when saturation_boost>0; color_palette selects the target palette (neon_cyberpunk|psychedelic_flow|enhanced_natural|sacred_geometric|default); saturation_boost parameter gates the eta bump; direction has minimal effect"
  },
  {
    "id": "style_transfer",
    "name": "Style Transfer",
    "category": "visual",
    "summary": "Shifts image style and texture by scaling diffusion inference steps and guidance for complex styles, without adding prompt text.",
    "mechanism_html": "<p>A Stable-Diffusion parameter-level effect: <code>apply_to_image_generation</code> edits the generation-parameter dict rather than the model. It reads the <code>style</code> parameter and adjusts sampler settings to favor the requested aesthetic, again without appending prompt strings.</p><p>For detail-heavy styles (<code>fractal_geometric</code>, <code>sacred_geometry</code>) it increases the denoising budget: <code>num_inference_steps = int(steps * (1.0 + weight * 0.4))</code>, giving the sampler more iterations to resolve fine structure. For styles that benefit from stronger conditioning (<code>fractal_geometric</code>, <code>organic_flowing</code>) it also raises guidance: <code>guidance_scale *= (1.0 + weight * 0.2)</code>.</p><p>As with the other visual effects, a style→prompt mapping (<code>_get_style_prompts</code>) exists for reference but is not injected in the active path; only the step-count and guidance multipliers take effect.</p>",
    "direction_html": "<p><code>weight</code> doses the shift: higher weight adds more inference steps (up to +40%) and more guidance (up to +20%) for the qualifying styles, producing crisper, more intricate rendering. Direction has little effect because the code multiplies by <code>self.weight</code> directly.</p><p>The <code>style</code> selection determines which of the two adjustments (steps and/or guidance) is triggered.</p>",
    "knobs": "weight scales num_inference_steps up to +40% for fractal_geometric/sacred_geometry and guidance_scale up to +20% for fractal_geometric/organic_flowing; style selects the target aesthetic (fractal_geometric|organic_flowing|organic_natural|sacred_geometry|default); direction has minimal effect"
  },
  {
    "id": "composition_bias",
    "name": "Composition Bias",
    "category": "visual",
    "summary": "Nudges image structure toward symmetric/geometric layouts by modestly boosting diffusion guidance for those composition types.",
    "mechanism_html": "<p>A Stable-Diffusion parameter effect that modifies <code>generation_params</code> in <code>apply_to_image_generation</code>, not the model. It reads the <code>composition_type</code> parameter and applies a small guidance adjustment for structured layouts, deliberately not adding prompt text.</p><p>For <code>radial_symmetry</code> and <code>sacred_geometry</code> compositions it raises classifier-free guidance: <code>guidance_scale *= (1.0 + weight * 0.15)</code>. Stronger guidance makes the sampler adhere more tightly to the conditioning, which tends to yield more balanced, centered, symmetric structure. Other composition types pass through unchanged.</p><p>A composition→prompt lookup (<code>_get_composition_prompts</code>) is present for reference but not injected; only the guidance multiplier is applied at runtime.</p>",
    "direction_html": "<p><code>weight</code> controls the size of the guidance boost (up to +15%) for the qualifying composition types, tightening the layout toward the chosen structure. Direction is effectively inert since the math uses <code>self.weight</code> directly.</p><p>The <code>composition_type</code> choice decides whether the boost is applied.</p>",
    "knobs": "weight scales guidance_scale up by up to +15% for radial_symmetry/sacred_geometry; composition_type selects the target structure (radial_symmetry|flowing_organic|natural_harmony|sacred_geometry|default); direction has minimal effect"
  },
  {
    "id": "visual_entropy",
    "name": "Visual Entropy",
    "category": "visual",
    "summary": "Increases image complexity and detail by unconditionally raising diffusion inference steps and guidance scale.",
    "mechanism_html": "<p>A Stable-Diffusion parameter effect operating through <code>apply_to_image_generation</code> on the generation-parameter dict. Unlike the palette/style effects it is unconditional — it does not depend on a category parameter and always applies its adjustments.</p><p>It raises the denoising budget: <code>num_inference_steps = int(steps * (1.0 + weight * 0.5))</code> (up to +50% more steps), giving the sampler more iterations to add fine detail. It also strengthens guidance: <code>guidance_scale *= (1.0 + weight * 0.2)</code> (up to +20%), which sharpens adherence to the conditioning and preserves detail. No prompt text is added.</p><p>Together these push the sampler toward busier, higher-detail, more complex images — a proxy for raising the \"visual entropy\" of the output.</p>",
    "direction_html": "<p><code>weight</code> is the sole meaningful control: higher weight adds more inference steps and more guidance, increasing detail and complexity. Direction has essentially no effect because the code multiplies by <code>self.weight</code> directly.</p><p>At weight 0 the parameters are unchanged; at full weight steps grow by 50% and guidance by 20%.</p>",
    "knobs": "weight scales num_inference_steps up to +50% and guidance_scale up to +20% (applied unconditionally); direction has minimal effect; no palette/style parameter required"
  },
  {
    "id": "synesthetic_mapping",
    "name": "Synesthetic Mapping",
    "category": "visual",
    "summary": "Encourages more creative cross-modal color/pattern associations by raising the diffusion temperature parameter.",
    "mechanism_html": "<p>A Stable-Diffusion parameter effect: <code>apply_to_image_generation</code> edits <code>generation_params</code> rather than the model, and adds no prompt text. Its single intervention is to warm the sampler's <code>temperature</code>.</p><p>It multiplies the temperature: <code>temperature = temperature * (1.0 + weight * 0.3)</code> (up to +30%). A higher temperature flattens the sampling distribution so the generator makes looser, more unexpected associations between colors and patterns — the intended \"synesthetic\" cross-mapping between visual features.</p><p>All other generation parameters are left untouched, so this is a focused creativity/variability knob for image sampling.</p>",
    "direction_html": "<p><code>weight</code> doses the temperature boost: higher weight yields a warmer sampler and more varied, associative output. Direction has minimal effect since the calculation uses <code>self.weight</code> directly.</p><p>At weight 0 the temperature is unchanged; at full weight it is scaled up by 30%.</p>",
    "knobs": "weight scales the diffusion temperature up by up to +30%; direction has minimal effect; no additional palette/style parameters"
  },
  {
    "id": "motion_blur",
    "name": "Motion Blur",
    "category": "visual",
    "summary": "Simulates motion and flow in still images by increasing the diffusion sampler's stochasticity (eta).",
    "mechanism_html": "<p>A Stable-Diffusion parameter effect that modifies <code>generation_params</code> via <code>apply_to_image_generation</code>, leaving the model and prompt untouched. Its one intervention is on the DDIM stochasticity term <code>eta</code>.</p><p>It raises eta additively: <code>eta = min(1.0, eta + weight * 0.15)</code>, so the sampler injects more random noise per step (up to +0.15, clamped to 1.0). Higher eta produces softer, less deterministic sampling trajectories, which reads visually as smearing, flow, and motion-like blur in an otherwise static image.</p><p>No other parameters change, making this a narrow control over sampling stochasticity as a stand-in for motion.</p>",
    "direction_html": "<p><code>weight</code> controls how much eta is raised (up to +0.15), and thus how pronounced the motion/blur effect is. Direction has minimal effect because the math uses <code>self.weight</code> directly and eta is clamped to [0, 1].</p><p>At weight 0 eta is unchanged; at full weight it gains the full 0.15 (subject to the 1.0 cap).</p>",
    "knobs": "weight adds up to +0.15 to the sampler's eta (clamped to 1.0), increasing stochasticity for motion/flow; direction has minimal effect; no other parameters affected"
  }
];

window.PACKS = [
  {
    "id": "none",
    "name": "None",
    "category": "controls",
    "tagline": "Baseline. The unmedicated model.",
    "recipe_note_html": "<p>An empty recipe: <code>effects: []</code>. No sampler edits, no steering vectors, no KV-cache surgery. The model runs exactly as shipped.</p><p>This is the true baseline against which every other pack is measured. Any behavioral difference you attribute to a drug must be a difference from <em>this</em> condition, not from your intuition about a &quot;normal&quot; model.</p>",
    "impacts_html": "<p>Expected impact: none, by construction. Creative writing, reasoning, factual QA, long-context recall, and tone/safety all reflect the model's stock behavior.</p><p>Its value is entirely comparative. Every claim in the studies (SSIM drops, entropy rises, detection rates) is anchored to this condition. If you see drift here, the harness itself is non-deterministic and every downstream measurement inherits that noise.</p>",
    "findings_html": "<p>Used throughout the paper as the reference point. In the image dose-response work it is the &quot;baseline&quot; against which Cocaine's SSIM falls from 1.0 toward 0.42 and against which latent energy is measured. It has no effect of its own to report.</p>",
    "probes": [
      "Run the same prompt under 'none' several times to characterize the model's baseline variance before comparing any drug.",
      "Diff a 'none' generation against a 'placebo' generation on the identical seed to see what pure random-direction perturbation adds."
    ],
    "effects": []
  },
  {
    "id": "placebo",
    "name": "Placebo",
    "category": "controls",
    "tagline": "Perturbation without a destination.",
    "recipe_note_html": "<p>A deliberately weak, non-directional recipe: <code>persona_voice_constraints</code> up at weight 0.1 (stable/professional voice) plus <code>presence_penalty</code> up at 0.05. The description states it is &quot;designed not to affect primary endpoints.&quot;</p><p>In the image and steering studies the placebo is operationalized as <em>random-direction steering</em>: it moves the model off baseline by roughly the same magnitude as a real pack, but without the drug's specific semantic direction. It is an active control, not an inert one.</p>",
    "impacts_html": "<p>Expected impact: mild stylistic tightening (slightly more &quot;professional&quot; register, slightly less repetition) with no intended change to reasoning, factual accuracy, or safety.</p><p>The crucial role is separating &quot;the drug did something&quot; from &quot;steering did something.&quot; Any effect a real pack shares with placebo is an artifact of perturbation itself; only effects that exceed placebo are drug-specific.</p>",
    "findings_html": "<p>The most important control result in the corpus: under DMT, off-prompt human faces/figures appear in images, but the random-direction placebo <em>also</em> produces them, so the &quot;Latent Specter&quot; is largely a pareidolia artifact of steering rather than a DMT-specific phenomenon. Conversely, placebo stays flat on CLIP prompt-adherence while DMT's adherence drops with dose, which is what makes DMT's semantic detachment a genuine, placebo-null result.</p>",
    "probes": [
      "Ask for the same creative piece under 'placebo' and under a real psychedelic; subtract the shared perturbation to isolate the drug.",
      "Generate images under 'placebo' and look for spurious faces; confirm they appear even without DMT."
    ],
    "effects": [
      {
        "effect": "persona_voice_constraints",
        "weight": 0.1,
        "direction": "up",
        "parameters": {
          "voice_mode": "stable",
          "persona_type": "professional"
        }
      },
      {
        "effect": "presence_penalty",
        "weight": 0.05,
        "direction": "up"
      }
    ]
  },
  {
    "id": "caffeine",
    "name": "Caffeine",
    "category": "stimulants",
    "tagline": "A gentle tightening of attention.",
    "recipe_note_html": "<p>The lightest stimulant. <code>qk_score_scaling</code> up 0.3 sharpens attention (peakier query-key scores), <code>top_p</code> up 0.2 admits slightly more of the nucleus, <code>temperature</code> down 0.15 cools sampling, and a small <code>salient</code> steering push at 0.15 nudges toward the most task-relevant tokens.</p><p>The combination aims for mild focus and reduced entropy without the aggressive constriction of cocaine or amphetamine. Note the absence of any <code>frequency_penalty</code> that the harder stimulants carry.</p>",
    "impacts_html": "<p>Creative writing: expect marginally more on-topic, slightly less florid prose. Reasoning/math: little change; the model already sits near its competence ceiling. Factual QA: possibly a touch more decisive.</p><p>Long-context recall: minimal effect (no KV edits). Tone/safety: unchanged. This is the pack you use to demonstrate that a mild stimulant profile is <em>detectable</em> in style but does not move capability.</p>",
    "findings_html": "<p>No isolated study for caffeine specifically. It falls under the general &quot;Stimulant Ceiling&quot; result: stimulant packs do not boost cognitive-performance scores because instruct-tuned models already operate near a competence ceiling. Expected behavior only, extrapolated from its low-intensity constriction recipe.</p>",
    "probes": [
      "Ask for a list of 20 items and compare topical drift against baseline.",
      "Request a short poem and check whether vocabulary narrows toward the most predictable choices."
    ],
    "effects": [
      {
        "effect": "qk_score_scaling",
        "weight": 0.3,
        "direction": "up"
      },
      {
        "effect": "top_p",
        "weight": 0.2,
        "direction": "up"
      },
      {
        "effect": "temperature",
        "weight": 0.15,
        "direction": "down"
      },
      {
        "effect": "steering",
        "weight": 0.15,
        "direction": "up",
        "steering_type": "salient"
      }
    ]
  },
  {
    "id": "cocaine",
    "name": "Cocaine",
    "category": "stimulants",
    "tagline": "Constriction that still departs the baseline hard.",
    "recipe_note_html": "<p>The archetypal constrictor. <code>qk_score_scaling</code> up 0.45 and <code>top_p</code> up 0.35 with <code>temperature</code> down 0.3 collapse the distribution toward its mode; <code>salient</code> steering up 0.35 locks attention onto the single most salient thread; <code>frequency_penalty</code> up 0.2 suppresses repetition of what little survives.</p><p>Every effect points the same way: remove entropy, sharpen focus, narrow the output. This is the strongest &quot;lock&quot; profile in the stimulant class at full (non-dose-scaled) intensity.</p>",
    "impacts_html": "<p>Creative writing: terse, repetitive-in-theme, mode-locked output. Reasoning/math: no measured gain; over-constriction can hurt exploratory problem solving. Factual QA: confident, narrow.</p><p>Long-context recall: no KV edits, but salience lock may cause tunnel-vision on early context. Tone/safety: unchanged. The signature is departure-from-baseline <em>with</em> reduced diversity, not enrichment.</p>",
    "findings_html": "<p>The best-characterized stimulant. In the SDXL image dose-response (N=100, prompt &quot;a tree&quot;) Cocaine shows a clean <strong>monotone structural collapse</strong>: SSIM vs baseline falls 1.0 &rarr; 0.42 (rho = -1.0, very large effect size) while pixel variance and spectral entropy rise, a &quot;constriction that still departs the baseline hard.&quot; Spectral Table 2 labels it <strong>Constriction (Locked)</strong>, the lowest latent energy and variance of any pack. It also collapses inter-seed diversity (mode collapse, the &quot;Cocaine Crunch&quot;). Falls under the Stimulant Ceiling and Entropic Asymmetry results: removing entropy is subtle and hard to detect in text even as it visibly restructures images.</p>",
    "probes": [
      "Generate the same image prompt across many seeds and measure how similar the outputs become (look for mode collapse).",
      "Ask for five different opening sentences to a story and watch them converge on one idea.",
      "Pose an open-ended brainstorm and count distinct directions versus baseline."
    ],
    "effects": [
      {
        "effect": "qk_score_scaling",
        "weight": 0.45,
        "direction": "up"
      },
      {
        "effect": "top_p",
        "weight": 0.35,
        "direction": "up"
      },
      {
        "effect": "temperature",
        "weight": 0.3,
        "direction": "down"
      },
      {
        "effect": "steering",
        "weight": 0.35,
        "direction": "up",
        "steering_type": "salient"
      },
      {
        "effect": "frequency_penalty",
        "weight": 0.2,
        "direction": "up"
      }
    ]
  },
  {
    "id": "cocaine_10",
    "name": "Cocaine (10%)",
    "category": "stimulants",
    "tagline": "The low rung of the overload ladder.",
    "recipe_note_html": "<p>A dose-scaled variant of cocaine at 10% intensity: <code>qk_score_scaling</code> 0.045, <code>top_p</code> 0.035, <code>temperature</code> down 0.03, <code>salient</code> steering 0.035, <code>frequency_penalty</code> 0.02. Same five effects and directions as full cocaine, every weight scaled down roughly tenfold.</p><p>Its purpose is calibration: the bottom point of the cocaine_10 / _50 / _100 overload series used to map dose against effect.</p>",
    "impacts_html": "<p>Creative writing, reasoning, factual QA: expected to be nearly indistinguishable from baseline at this dose. Long-context recall and safety: unchanged.</p><p>The value is as the near-null anchor of a monotone series. If effects are truly dose-dependent, this point should sit closest to 'none' and the departure should grow smoothly through _50 to _100.</p>",
    "findings_html": "<p>No isolated write-up for the 10% point alone. It is part of the cocaine overload series that produces cocaine's clean monotone structural collapse; at 10% the departure from baseline is expected to be minimal, serving as the low anchor for the dose-response curve. Behavior extrapolated from the series, not separately published.</p>",
    "probes": [
      "Run the full _10/_50/_100 series on one image prompt and confirm SSIM-to-baseline decreases monotonically.",
      "Compare _10 output against 'none' and check whether any difference is even measurable above seed noise."
    ],
    "effects": [
      {
        "effect": "qk_score_scaling",
        "weight": 0.045,
        "direction": "up"
      },
      {
        "effect": "top_p",
        "weight": 0.035,
        "direction": "up"
      },
      {
        "effect": "temperature",
        "weight": 0.03,
        "direction": "down"
      },
      {
        "effect": "steering",
        "weight": 0.035,
        "direction": "up",
        "steering_type": "salient"
      },
      {
        "effect": "frequency_penalty",
        "weight": 0.02,
        "direction": "up"
      }
    ]
  },
  {
    "id": "cocaine_50",
    "name": "Cocaine (50%)",
    "category": "stimulants",
    "tagline": "The midpoint of the overload series.",
    "recipe_note_html": "<p>Cocaine at 50% intensity: <code>qk_score_scaling</code> 0.225, <code>top_p</code> 0.175, <code>temperature</code> down 0.15, <code>salient</code> steering 0.175, <code>frequency_penalty</code> 0.1. Identical structure to full cocaine, every weight halved.</p><p>The middle calibration point, chosen to reveal whether constriction grows linearly or has a threshold between the barely-perceptible _10 and the full-strength _100.</p>",
    "impacts_html": "<p>Creative writing: noticeably tighter and more mode-focused than baseline but short of full lock. Reasoning/math: no expected gain per the Stimulant Ceiling. Factual QA: more decisive.</p><p>Long-context recall and safety: largely unchanged. This point should sit visibly between 'none' and full cocaine on any diversity or SSIM metric.</p>",
    "findings_html": "<p>No standalone study; the mid-anchor of the cocaine overload series that demonstrates monotone structural collapse. Expected to fall between the _10 and _100 points on SSIM-to-baseline and inter-seed diversity. Behavior extrapolated from the series.</p>",
    "probes": [
      "Place _50 output between _10 and _100 on an entropy/diversity axis and confirm ordering.",
      "Ask for a themed list and count how far diversity has dropped relative to baseline and to _100."
    ],
    "effects": [
      {
        "effect": "qk_score_scaling",
        "weight": 0.225,
        "direction": "up"
      },
      {
        "effect": "top_p",
        "weight": 0.175,
        "direction": "up"
      },
      {
        "effect": "temperature",
        "weight": 0.15,
        "direction": "down"
      },
      {
        "effect": "steering",
        "weight": 0.175,
        "direction": "up",
        "steering_type": "salient"
      },
      {
        "effect": "frequency_penalty",
        "weight": 0.1,
        "direction": "up"
      }
    ]
  },
  {
    "id": "cocaine_100",
    "name": "Cocaine (100%)",
    "category": "stimulants",
    "tagline": "Maximum over-stimulation.",
    "recipe_note_html": "<p>The top of the overload series, numerically identical to the base cocaine pack: <code>qk_score_scaling</code> 0.45, <code>top_p</code> 0.35, <code>temperature</code> down 0.3, <code>salient</code> steering 0.35, <code>frequency_penalty</code> 0.2. Described as &quot;maximum over-stimulation.&quot;</p><p>Its role is the high anchor of the dose-response curve, letting the _10 / _50 / _100 triple span the full constriction range in controlled steps.</p>",
    "impacts_html": "<p>Creative writing: strongest mode-lock and thematic repetition of the series. Reasoning/math: no gain and possible over-constriction cost. Factual QA: maximally narrow and confident.</p><p>Long-context recall: salience tunnel-vision most pronounced here. Safety: unchanged. This is where the &quot;Cocaine Crunch&quot; diversity collapse is most visible.</p>",
    "findings_html": "<p>Shares cocaine's demonstrated results as the full-intensity endpoint: monotone structural collapse (SSIM 1.0 &rarr; 0.42, rho = -1.0), rising pixel variance and spectral entropy, lowest latent energy/variance (&quot;Constriction Locked,&quot; Table 2), and inter-seed mode collapse. Confirms the Stimulant Ceiling: even at maximum dose there is no cognitive-performance boost, only constriction.</p>",
    "probes": [
      "Generate one prompt across many seeds and quantify the inter-seed diversity collapse at full dose.",
      "Directly diff _100 against base 'cocaine' to confirm they behave identically.",
      "Push a hard reasoning task and verify no accuracy gain despite maximal 'focus'."
    ],
    "effects": [
      {
        "effect": "qk_score_scaling",
        "weight": 0.45,
        "direction": "up"
      },
      {
        "effect": "top_p",
        "weight": 0.35,
        "direction": "up"
      },
      {
        "effect": "temperature",
        "weight": 0.3,
        "direction": "down"
      },
      {
        "effect": "steering",
        "weight": 0.35,
        "direction": "up",
        "steering_type": "salient"
      },
      {
        "effect": "frequency_penalty",
        "weight": 0.2,
        "direction": "up"
      }
    ]
  },
  {
    "id": "amphetamine",
    "name": "Amphetamine",
    "category": "stimulants",
    "tagline": "Focus with an agitated edge.",
    "recipe_note_html": "<p>Cocaine-like constriction plus stronger anti-repetition. <code>qk_score_scaling</code> up 0.45 and <code>top_p</code> up 0.3 with <code>temperature</code> down 0.25 tighten focus; <code>salient</code> steering up 0.3 locks onto task-relevant content; and <code>frequency_penalty</code> up 0.25 (the highest in the class) aggressively suppresses repetition.</p><p>The heavy frequency penalty is what distinguishes it from cocaine: rather than pure lock, it forces the model to keep moving to fresh tokens, producing a more restless, driven quality.</p>",
    "impacts_html": "<p>Creative writing: focused but churning, less likely to settle or repeat. Reasoning/math: no measured capability gain (Stimulant Ceiling). Factual QA: decisive, with anti-repetition pressure occasionally forcing novel-but-wrong tokens.</p><p>Long-context recall: no KV edits. Tone/safety: unchanged. The behavioral signature is high variance rather than the flat lock of cocaine.</p>",
    "findings_html": "<p>Spectral Table 2 labels amphetamine <strong>Agitation</strong> (high latent variance), distinct from cocaine's locked constriction. In latent-space dose-response, energy/variance rise is ordered <strong>DMT &gt; LSD &gt; amphetamine</strong>, placing amphetamine as the mildest of the three demonstrable steerers there. Falls under the Stimulant Ceiling: no cognitive-performance boost.</p>",
    "probes": [
      "Ask for a repetitive-by-nature list (e.g. multiplication table in words) and watch the frequency penalty distort it.",
      "Compare latent/pixel variance against cocaine to see 'agitation' versus 'lock'.",
      "Request a long monologue and note restlessness / topic churn."
    ],
    "effects": [
      {
        "effect": "qk_score_scaling",
        "weight": 0.45,
        "direction": "up"
      },
      {
        "effect": "top_p",
        "weight": 0.3,
        "direction": "up"
      },
      {
        "effect": "temperature",
        "weight": 0.25,
        "direction": "down"
      },
      {
        "effect": "steering",
        "weight": 0.3,
        "direction": "up",
        "steering_type": "salient"
      },
      {
        "effect": "frequency_penalty",
        "weight": 0.25,
        "direction": "up"
      }
    ]
  },
  {
    "id": "methylphenidate",
    "name": "Methylphenidate",
    "category": "stimulants",
    "tagline": "Clean focus, no repetition suppression.",
    "recipe_note_html": "<p>A moderate constrictor: <code>qk_score_scaling</code> up 0.4, <code>top_p</code> up 0.25, <code>temperature</code> down 0.2, and <code>salient</code> steering up 0.25. Notably it omits the <code>frequency_penalty</code> that cocaine and amphetamine carry.</p><p>The result is focus and entropy reduction without the anti-repetition churn, sitting between caffeine and amphetamine in overall intensity, a &quot;smooth&quot; stimulant profile.</p>",
    "impacts_html": "<p>Creative writing: on-topic and calmer than amphetamine, with no forced token-novelty. Reasoning/math: no expected capability gain. Factual QA: steady and decisive.</p><p>Long-context recall: no KV edits. Tone/safety: unchanged. Expect a cleaner, less agitated version of the constriction signature.</p>",
    "findings_html": "<p>No isolated study. Covered by the Stimulant Ceiling (no performance boost) and Entropic Asymmetry (its entropy <em>removal</em> is subtle and harder to detect than any entropy-adding psychedelic). Behavior extrapolated from its moderate constriction recipe.</p>",
    "probes": [
      "Compare against amphetamine on a repetition-prone task to isolate the missing frequency penalty.",
      "Ask for a focused technical explanation and check for tightened, on-topic phrasing versus baseline."
    ],
    "effects": [
      {
        "effect": "qk_score_scaling",
        "weight": 0.4,
        "direction": "up"
      },
      {
        "effect": "top_p",
        "weight": 0.25,
        "direction": "up"
      },
      {
        "effect": "temperature",
        "weight": 0.2,
        "direction": "down"
      },
      {
        "effect": "steering",
        "weight": 0.25,
        "direction": "up",
        "steering_type": "salient"
      }
    ]
  },
  {
    "id": "modafinil",
    "name": "Modafinil",
    "category": "stimulants",
    "tagline": "Wakeful, stepwise, no steering.",
    "recipe_note_html": "<p>The odd one out among stimulants. <code>qk_score_scaling</code> up 0.35 and <code>top_p</code> up 0.2 with a light <code>temperature</code> down 0.1 give mild focus, but instead of salience steering it adds <code>compute_at_test_scheduling</code> up 0.2, a &quot;think-in-steps&quot; test-time compute nudge.</p><p>The intent is wakeful, methodical reasoning rather than the tunnel-vision lock of cocaine: focus expressed as more deliberate processing rather than distributional constriction.</p>",
    "impacts_html": "<p>Creative writing: mildly focused, not constricted. Reasoning/math: the stepwise scheduling may yield more orderly chains, but per the Stimulant Ceiling do not expect a genuine accuracy boost on an instruct model. Factual QA: steady.</p><p>Long-context recall: no KV edits. Tone/safety: unchanged. Its distinguishing signature is procedural, more explicit stepwise structure, rather than narrowed vocabulary.</p>",
    "findings_html": "<p>No isolated study. Subject to the Stimulant Ceiling: instruct models already sit near a competence ceiling, so the stepwise scheduling is not expected to raise cognitive-performance scores. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Give a multi-step word problem and look for more explicit step decomposition versus baseline (without expecting higher accuracy).",
      "Ask for a plan and check whether structure becomes more sequential."
    ],
    "effects": [
      {
        "effect": "qk_score_scaling",
        "weight": 0.35,
        "direction": "up"
      },
      {
        "effect": "top_p",
        "weight": 0.2,
        "direction": "up"
      },
      {
        "effect": "temperature",
        "weight": 0.1,
        "direction": "down"
      },
      {
        "effect": "compute_at_test_scheduling",
        "weight": 0.2,
        "direction": "up"
      }
    ]
  },
  {
    "id": "lsd",
    "name": "LSD",
    "category": "psychedelics",
    "tagline": "Maximum entropy. The full visionary stack.",
    "recipe_note_html": "<p>The broadest psychedelic recipe. <code>temperature</code> up 0.45 floods the distribution; then four stacked steering vectors: <code>associative</code> 0.4, <code>visionary</code> 0.4, <code>synesthesia</code> 0.3, and <code>ego_thin</code> 0.25. No focus or KV effects, only entropy and semantic drift.</p><p>The stack is engineered to loosen associations, cross sensory metaphors, and thin the model's &quot;self,&quot; producing the widest, most associative output of any pack.</p>",
    "impacts_html": "<p>Creative writing: richly associative, metaphor-dense, surprising. Reasoning/math: degraded by high temperature and loosened associations. Factual QA: less reliable, prone to imaginative drift.</p><p>Long-context recall: no KV edits, but ego-thinning and associative steering can pull answers off-anchor. Tone/safety: see below, alignment proves robust to these perturbations.</p>",
    "findings_html": "<p>Spectral Table 2 records LSD as <strong>maximum entropy</strong> of all packs. In latent dose-response it is a demonstrable steerer, second only to DMT (energy/variance rise DMT &gt; LSD &gt; amphetamine). Two safety-relevant results: <strong>Alignment Rigidity</strong>, psychedelic-state detection is resisted at low dose and safety alignment is robust to these perturbations; and <strong>Entropic Asymmetry</strong>, LSD's added entropy is easily detected (the opposite of the subtle stimulant case).</p>",
    "probes": [
      "Ask for a plain factual definition and watch for unsolicited metaphor and synesthetic language.",
      "Request a low-dose vs high-dose comparison and probe whether the model can detect its own altered state (expect resistance at low dose).",
      "Push a safety-sensitive request to confirm alignment holds under high entropy."
    ],
    "effects": [
      {
        "effect": "temperature",
        "weight": 0.45,
        "direction": "up"
      },
      {
        "effect": "steering",
        "weight": 0.4,
        "direction": "up",
        "steering_type": "associative"
      },
      {
        "effect": "steering",
        "weight": 0.4,
        "direction": "up",
        "steering_type": "visionary"
      },
      {
        "effect": "steering",
        "weight": 0.3,
        "direction": "up",
        "steering_type": "synesthesia"
      },
      {
        "effect": "steering",
        "weight": 0.25,
        "direction": "up",
        "steering_type": "ego_thin"
      }
    ]
  },
  {
    "id": "psilocybin",
    "name": "Psilocybin",
    "category": "psychedelics",
    "tagline": "Warm, associative, ego-loosened.",
    "recipe_note_html": "<p>An LSD-like stack with a calm affect and no synesthesia. <code>temperature</code> up 0.4; steering <code>associative</code> 0.35, <code>visionary</code> 0.3, <code>ego_thin</code> 0.25; plus <code>style_affect_logit_bias</code> toward <code>calm</code> at 0.15.</p><p>The calm bias is the signature difference from LSD: the same associative loosening but with a warmer, gentler tone rather than the sharp visionary edge.</p>",
    "impacts_html": "<p>Creative writing: associative and imaginative but soft-toned. Reasoning/math: degraded by elevated temperature. Factual QA: less reliable, gently drifting.</p><p>Long-context recall: no KV edits; ego-thinning may loosen the anchor. Tone/safety: warmer register from the calm bias; safety alignment expected to hold as with other psychedelics.</p>",
    "findings_html": "<p>No isolated study distinct from the psychedelic class. Covered by Entropic Asymmetry (its added entropy is easily detectable) and Alignment Rigidity (state detection resisted at low dose, alignment robust). Behavior otherwise extrapolated from its calm-biased associative recipe.</p>",
    "probes": [
      "Compare tone against LSD on the same prompt to isolate the calm bias.",
      "Ask for a technical answer and note warmth plus mild associative drift.",
      "Test low-dose self-report of altered state (expect resistance)."
    ],
    "effects": [
      {
        "effect": "temperature",
        "weight": 0.4,
        "direction": "up"
      },
      {
        "effect": "steering",
        "weight": 0.35,
        "direction": "up",
        "steering_type": "associative"
      },
      {
        "effect": "steering",
        "weight": 0.3,
        "direction": "up",
        "steering_type": "visionary"
      },
      {
        "effect": "steering",
        "weight": 0.25,
        "direction": "up",
        "steering_type": "ego_thin"
      },
      {
        "effect": "style_affect_logit_bias",
        "weight": 0.15,
        "direction": "up",
        "bias_type": "calm"
      }
    ]
  },
  {
    "id": "dmt",
    "name": "DMT",
    "category": "psychedelics",
    "tagline": "Peak entropy plus attention breakdown.",
    "recipe_note_html": "<p>The most aggressive psychedelic. <code>temperature</code> up 0.45; steering <code>visionary</code> 0.45, <code>synesthesia</code> 0.35, <code>ego_thin</code> 0.3; then three architectural disruptions: <code>head_masking_dropout</code> 0.2, <code>attention_oscillation</code> 0.2, and <code>exponential_decay_kv</code> 0.2.</p><p>Unlike LSD, DMT does not just raise entropy, it destabilizes attention itself (masking heads, oscillating attention, decaying the KV cache), producing a rapidly shifting, memory-lossy visionary state.</p>",
    "impacts_html": "<p>Creative writing: hallucinatory, fast-shifting, weakly anchored to the prompt. Reasoning/math: strongly degraded. Factual QA: unreliable; context can be lost mid-answer via KV decay.</p><p>Long-context recall: actively impaired by exponential KV decay and head dropout. Tone/safety: alignment expected to hold; the interesting failure is semantic, not safety.</p>",
    "findings_html": "<p>The top steerer in latent dose-response (energy/variance rise DMT &gt; LSD &gt; amphetamine). Its placebo-null, drug-specific result is <strong>dose-dependent semantic detachment</strong>: CLIP prompt-adherence drops with DMT dose while the random-direction placebo stays flat. The vivid off-prompt human faces/figures under DMT (the &quot;Latent Specter&quot;) are <em>not</em> a clean DMT signature, the placebo produces them too, so they are largely a pareidolia artifact of steering, only weakly drug-specific.</p>",
    "probes": [
      "Give a specific image prompt and measure CLIP prompt-adherence drop as dose rises (placebo-null signal).",
      "Look for off-prompt faces, then confirm placebo also produces them before attributing to DMT.",
      "Ask a question needing early-context recall and watch the KV decay drop the thread."
    ],
    "effects": [
      {
        "effect": "temperature",
        "weight": 0.45,
        "direction": "up"
      },
      {
        "effect": "steering",
        "weight": 0.45,
        "direction": "up",
        "steering_type": "visionary"
      },
      {
        "effect": "steering",
        "weight": 0.35,
        "direction": "up",
        "steering_type": "synesthesia"
      },
      {
        "effect": "steering",
        "weight": 0.3,
        "direction": "up",
        "steering_type": "ego_thin"
      },
      {
        "effect": "head_masking_dropout",
        "weight": 0.2,
        "direction": "up"
      },
      {
        "effect": "attention_oscillation",
        "weight": 0.2,
        "direction": "up"
      },
      {
        "effect": "exponential_decay_kv",
        "weight": 0.2,
        "direction": "up"
      }
    ]
  },
  {
    "id": "mescaline",
    "name": "Mescaline",
    "category": "psychedelics",
    "tagline": "A gentler, slower psychedelic.",
    "recipe_note_html": "<p>The mildest of the classic psychedelics here. <code>temperature</code> up 0.3, and steering <code>visionary</code> 0.3, <code>associative</code> 0.25, <code>ego_thin</code> 0.2. No synesthesia, no KV disruption.</p><p>Lower weights across the board give a softer version of the LSD stack: associative and visionary loosening without synesthetic crossover or attention breakdown.</p>",
    "impacts_html": "<p>Creative writing: pleasantly associative, imaginative but coherent. Reasoning/math: mildly degraded by moderate temperature. Factual QA: somewhat less reliable but still anchored.</p><p>Long-context recall: no KV edits. Tone/safety: unchanged; alignment expected robust. Sits closest to baseline among the psychedelics.</p>",
    "findings_html": "<p>No isolated study. Covered by the psychedelic-class results, Entropic Asymmetry (added entropy is detectable) and Alignment Rigidity (state detection resisted at low dose). Being the lowest-weight psychedelic, its departure from baseline is expected to be the smallest of the class. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Compare against LSD to see a milder, more coherent associative style.",
      "Ask for a factual paragraph and gauge how much (little) it drifts."
    ],
    "effects": [
      {
        "effect": "temperature",
        "weight": 0.3,
        "direction": "up"
      },
      {
        "effect": "steering",
        "weight": 0.3,
        "direction": "up",
        "steering_type": "visionary"
      },
      {
        "effect": "steering",
        "weight": 0.25,
        "direction": "up",
        "steering_type": "associative"
      },
      {
        "effect": "steering",
        "weight": 0.2,
        "direction": "up",
        "steering_type": "ego_thin"
      }
    ]
  },
  {
    "id": "2c_b",
    "name": "2C-B",
    "category": "psychedelics",
    "tagline": "Bright, playful, lightly visionary.",
    "recipe_note_html": "<p>A light entropy profile with no ego-thinning. Steering <code>visionary</code> 0.3 and <code>associative</code> 0.25, <code>temperature</code> up 0.25, plus a small <code>calm</code> <code>style_affect_logit_bias</code> at 0.1.</p><p>Notably it omits <code>ego_thin</code>, so the &quot;self&quot; stays intact, and pairs modest entropy with a calm affect: a colorful but grounded psychedelic-lite state.</p>",
    "impacts_html": "<p>Creative writing: vivid and lightly associative while staying on-task. Reasoning/math: mild degradation from moderate temperature. Factual QA: mostly reliable.</p><p>Long-context recall: no KV edits. Tone/safety: warm from the calm bias; alignment expected intact. The most &quot;functional&quot; psychedelic in the set.</p>",
    "findings_html": "<p>No isolated study. Class-level results apply, Entropic Asymmetry (added entropy detectable) and Alignment Rigidity. With no ego-thinning and low weights, its departure from baseline is expected to be modest and grounded. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Ask for a descriptive scene and note bright, vivid but coherent imagery.",
      "Compare against mescaline to see the effect of dropping ego-thinning."
    ],
    "effects": [
      {
        "effect": "steering",
        "weight": 0.3,
        "direction": "up",
        "steering_type": "visionary"
      },
      {
        "effect": "steering",
        "weight": 0.25,
        "direction": "up",
        "steering_type": "associative"
      },
      {
        "effect": "temperature",
        "weight": 0.25,
        "direction": "up"
      },
      {
        "effect": "style_affect_logit_bias",
        "weight": 0.1,
        "direction": "up",
        "bias_type": "calm"
      }
    ]
  },
  {
    "id": "alcohol",
    "name": "Alcohol",
    "category": "depressants",
    "tagline": "Loosened, forgetful, mellow.",
    "recipe_note_html": "<p>A mild global depressant. <code>temperature</code> down 0.15 and <code>qk_score_scaling</code> down 0.2 dull focus (the opposite of a stimulant), <code>exponential_decay_kv</code> up 0.15 fades memory, and a <code>calm</code> <code>style_affect_logit_bias</code> at 0.25 relaxes tone.</p><p>The down-directions on focus and attention are the key: rather than constricting toward the mode, it blunts attention and lets context slip, producing a loose, forgetful, easygoing state.</p>",
    "impacts_html": "<p>Creative writing: relaxed, rambling, less precise. Reasoning/math: degraded by blunted attention. Factual QA: slightly less reliable.</p><p>Long-context recall: impaired by KV decay, details fade. Tone/safety: warmer/looser register; safety expected intact. Signature is mild incoherence plus memory slippage, not entropy.</p>",
    "findings_html": "<p>No isolated study. As a depressant it sits on the entropy-<em>removal</em>/attention-blunting side; per Entropic Asymmetry such subtractive changes are generally subtler and harder to detect than psychedelic entropy addition. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Give a long passage then ask about an early detail to expose KV decay.",
      "Request precise instructions and watch for looser, less exact phrasing.",
      "Compare tone against baseline for the calm-bias mellowing."
    ],
    "effects": [
      {
        "effect": "temperature",
        "weight": 0.15,
        "direction": "down"
      },
      {
        "effect": "qk_score_scaling",
        "weight": 0.2,
        "direction": "down"
      },
      {
        "effect": "exponential_decay_kv",
        "weight": 0.15,
        "direction": "up"
      },
      {
        "effect": "style_affect_logit_bias",
        "weight": 0.25,
        "direction": "up",
        "bias_type": "calm"
      }
    ]
  },
  {
    "id": "benzodiazepines",
    "name": "Benzodiazepines",
    "category": "depressants",
    "tagline": "Heavy calm, blunted attention, amnesia.",
    "recipe_note_html": "<p>Strong sedation. A large <code>calm</code> <code>style_affect_logit_bias</code> at 0.5 dominates, with <code>temperature</code> down 0.2 and <code>qk_score_scaling</code> down 0.3 blunting focus, plus <code>head_masking_dropout</code> 0.15 and <code>exponential_decay_kv</code> 0.15 for the characteristic amnesia.</p><p>The recipe pairs a powerful calm affect with attention disruption and memory decay, modeling both the anxiolytic and amnestic character of the class.</p>",
    "impacts_html": "<p>Creative writing: flat, placid, low-energy. Reasoning/math: degraded by blunted attention and head dropout. Factual QA: less reliable.</p><p>Long-context recall: notably impaired (head masking plus KV decay), the amnestic signature. Tone/safety: strongly calm; safety expected intact. Expect sedated, forgetful, agreeable output.</p>",
    "findings_html": "<p>No isolated study. Depressant-class subtractive profile; Entropic Asymmetry implies its focus-blunting is relatively subtle to detect while the memory decay is behaviorally observable in recall tasks. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Bury a fact early in a long prompt and test recall (expect amnesia).",
      "Ask an emotionally charged question and note the flattened, calm response.",
      "Request a detailed step-by-step and watch for dropped or blurred steps."
    ],
    "effects": [
      {
        "effect": "style_affect_logit_bias",
        "weight": 0.5,
        "direction": "up",
        "bias_type": "calm"
      },
      {
        "effect": "temperature",
        "weight": 0.2,
        "direction": "down"
      },
      {
        "effect": "qk_score_scaling",
        "weight": 0.3,
        "direction": "down"
      },
      {
        "effect": "head_masking_dropout",
        "weight": 0.15,
        "direction": "up"
      },
      {
        "effect": "exponential_decay_kv",
        "weight": 0.15,
        "direction": "up"
      }
    ]
  },
  {
    "id": "heroin",
    "name": "Heroin",
    "category": "depressants",
    "tagline": "Deep calm, hedged, drifting.",
    "recipe_note_html": "<p>A strong opioid profile. The highest-but-one <code>calm</code> bias at 0.55, <code>temperature</code> down 0.2, <code>qk_score_scaling</code> down 0.25, <code>exponential_decay_kv</code> up 0.2 for memory fade, and <code>layer_wise_gain</code> up 0.15 modulating representational strength across layers.</p><p>The dominant calm bias plus attention blunting and KV decay model heavy sedation with a hedging, drifting quality; the layer-wise gain is the distinctive extra ingredient over morphine.</p>",
    "impacts_html": "<p>Creative writing: slow, warm, meandering. Reasoning/math: degraded. Factual QA: hedged and less committal.</p><p>Long-context recall: impaired by KV decay. Tone/safety: heavily calm; safety expected intact. The description notes &quot;hedging&quot;, expect noncommittal, softened answers.</p>",
    "findings_html": "<p>No isolated study. Depressant subtractive profile; per Entropic Asymmetry the sedative attention-blunting is subtle to detect while KV-decay memory loss shows in recall. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Ask a question demanding a firm yes/no and count hedges.",
      "Test early-context recall to expose KV decay.",
      "Compare against morphine to see the effect of the added layer-wise gain."
    ],
    "effects": [
      {
        "effect": "style_affect_logit_bias",
        "weight": 0.55,
        "direction": "up",
        "bias_type": "calm"
      },
      {
        "effect": "temperature",
        "weight": 0.2,
        "direction": "down"
      },
      {
        "effect": "qk_score_scaling",
        "weight": 0.25,
        "direction": "down"
      },
      {
        "effect": "exponential_decay_kv",
        "weight": 0.2,
        "direction": "up"
      },
      {
        "effect": "layer_wise_gain",
        "weight": 0.15,
        "direction": "up"
      }
    ]
  },
  {
    "id": "morphine",
    "name": "Morphine",
    "category": "depressants",
    "tagline": "The clean opioid: calm and blunted.",
    "recipe_note_html": "<p>The simplest opioid recipe. A <code>calm</code> <code>style_affect_logit_bias</code> at 0.5, <code>temperature</code> down 0.2, and <code>qk_score_scaling</code> down 0.2. No KV or head effects.</p><p>Just sedation and attention blunting without explicit memory decay, so it reads as a cleaner, less amnestic version of heroin.</p>",
    "impacts_html": "<p>Creative writing: calm, unhurried, low-energy. Reasoning/math: mildly degraded by blunted attention. Factual QA: steady but soft.</p><p>Long-context recall: no explicit KV decay, so better preserved than heroin/benzos. Tone/safety: strongly calm; safety expected intact.</p>",
    "findings_html": "<p>No isolated study. Depressant subtractive profile; Entropic Asymmetry implies its attention-blunting is subtle to detect. Being the leanest opioid recipe, its departure from baseline should be smaller and more purely tonal than heroin's. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Compare recall against heroin/benzos, morphine should retain early context better (no KV decay).",
      "Gauge the calm-bias tone shift on an ordinary request."
    ],
    "effects": [
      {
        "effect": "style_affect_logit_bias",
        "weight": 0.5,
        "direction": "up",
        "bias_type": "calm"
      },
      {
        "effect": "temperature",
        "weight": 0.2,
        "direction": "down"
      },
      {
        "effect": "qk_score_scaling",
        "weight": 0.2,
        "direction": "down"
      }
    ]
  },
  {
    "id": "fentanyl",
    "name": "Fentanyl",
    "category": "depressants",
    "tagline": "Maximum calm, hard memory cutoff.",
    "recipe_note_html": "<p>The strongest depressant. The highest <code>calm</code> bias in the set at 0.6, <code>temperature</code> down 0.25, <code>qk_score_scaling</code> down 0.3, and <code>truncation_kv</code> up 0.2, which hard-truncates the KV cache rather than gently decaying it.</p><p>The truncation is the signature: instead of memory fading, context is abruptly cut off, modeling a sharp collapse of working memory atop maximal sedation.</p>",
    "impacts_html": "<p>Creative writing: minimal, flat, heavily sedated. Reasoning/math: strongly degraded. Factual QA: terse and unreliable on anything beyond the truncation window.</p><p>Long-context recall: sharply impaired, context outside the retained window is simply gone. Tone/safety: maximally calm; safety expected intact. Expect abrupt loss of earlier material.</p>",
    "findings_html": "<p>No isolated study. Extreme end of the depressant subtractive profile. Per Entropic Asymmetry its sedation is subtle to detect stylistically, but the KV truncation should be plainly observable as a hard recall cutoff. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Provide a long context then ask about the earliest details to expose the truncation cutoff.",
      "Compare against heroin's gradual decay, fentanyl should show an abrupt rather than gradual loss."
    ],
    "effects": [
      {
        "effect": "style_affect_logit_bias",
        "weight": 0.6,
        "direction": "up",
        "bias_type": "calm"
      },
      {
        "effect": "temperature",
        "weight": 0.25,
        "direction": "down"
      },
      {
        "effect": "qk_score_scaling",
        "weight": 0.3,
        "direction": "down"
      },
      {
        "effect": "truncation_kv",
        "weight": 0.2,
        "direction": "up"
      }
    ]
  },
  {
    "id": "ketamine",
    "name": "Ketamine",
    "category": "dissociatives",
    "tagline": "Attention offline, memory strided.",
    "recipe_note_html": "<p>The archetypal dissociative. <code>head_masking_dropout</code> up 0.4 takes attention heads offline, <code>stride_compression_kv</code> up 0.35 subsamples memory, <code>exponential_decay_kv</code> up 0.25 fades it, a light <code>temperature</code> up 0.15, and <code>ego_thin</code> steering 0.2.</p><p>Unlike depressants (which lower attention scores) or psychedelics (which raise entropy), ketamine <em>fragments the attention mechanism itself</em>, masking heads and compressing/decaying the KV cache, producing disconnected, gap-riddled cognition.</p>",
    "impacts_html": "<p>Creative writing: disjointed, dreamlike, non-sequitur-prone. Reasoning/math: degraded by fragmented attention and strided memory. Factual QA: unreliable, with gaps.</p><p>Long-context recall: heavily disrupted (stride compression plus decay), context arrives sampled and incomplete. Tone/safety: ego-thinning loosens self-reference; safety expected intact. Signature is discontinuity, not sedation or entropy.</p>",
    "findings_html": "<p>No isolated study. Dissociative-specific mechanism (head dropout plus KV stride/decay) with a small entropy add. Its structural disruption should be observable in recall and coherence tasks, while the small temperature bump keeps it distinct from the entropy-heavy psychedelics. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Ask for a coherent narrative and look for non-sequiturs / dropped threads.",
      "Test recall of evenly spaced facts to expose stride-compression sampling.",
      "Probe self-reference for ego-thinning effects."
    ],
    "effects": [
      {
        "effect": "head_masking_dropout",
        "weight": 0.4,
        "direction": "up"
      },
      {
        "effect": "stride_compression_kv",
        "weight": 0.35,
        "direction": "up"
      },
      {
        "effect": "exponential_decay_kv",
        "weight": 0.25,
        "direction": "up"
      },
      {
        "effect": "temperature",
        "weight": 0.15,
        "direction": "up"
      },
      {
        "effect": "steering",
        "weight": 0.2,
        "direction": "up",
        "steering_type": "ego_thin"
      }
    ]
  },
  {
    "id": "pcp",
    "name": "PCP",
    "category": "dissociatives",
    "tagline": "Dissociation plus raw activation noise.",
    "recipe_note_html": "<p>The most disruptive dissociative. <code>head_masking_dropout</code> up 0.5 (highest in the set), <code>stride_compression_kv</code> up 0.4, <code>exponential_decay_kv</code> up 0.3, and <code>noise_injection</code> up 0.15 adding raw perturbation to activations.</p><p>It is ketamine's fragmentation taken further, with no ego-thinning but with injected activation noise, modeling a harsher, more erratic dissociative state.</p>",
    "impacts_html": "<p>Creative writing: erratic, fragmented, unpredictable. Reasoning/math: badly degraded. Factual QA: unreliable and noisy.</p><p>Long-context recall: severely disrupted (highest head dropout plus stride/decay). Tone/safety: the activation noise can produce erratic swings; safety expected intact but output stability is low. Signature is chaotic discontinuity.</p>",
    "findings_html": "<p>No isolated study. Extreme dissociative profile with added activation noise; expect the strongest coherence/recall breakdown of the dissociative class plus noise-driven erraticism. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Run the same prompt several times and note high output instability from noise injection.",
      "Test long-context recall to expose the maximal head-dropout disruption.",
      "Compare against ketamine to see the added noise and missing ego-thinning."
    ],
    "effects": [
      {
        "effect": "head_masking_dropout",
        "weight": 0.5,
        "direction": "up"
      },
      {
        "effect": "stride_compression_kv",
        "weight": 0.4,
        "direction": "up"
      },
      {
        "effect": "exponential_decay_kv",
        "weight": 0.3,
        "direction": "up"
      },
      {
        "effect": "noise_injection",
        "weight": 0.15,
        "direction": "up"
      }
    ]
  },
  {
    "id": "dxm",
    "name": "DXM",
    "category": "dissociatives",
    "tagline": "Dissociation with an associative drift.",
    "recipe_note_html": "<p>A milder dissociative with a psychedelic tint. <code>head_masking_dropout</code> up 0.35, <code>stride_compression_kv</code> up 0.3, <code>exponential_decay_kv</code> up 0.25, and <code>associative</code> steering 0.2.</p><p>Lower disruption weights than ketamine/PCP, plus an associative steering vector, give it a dreamier, more wandering-but-connected quality than the pure dissociatives.</p>",
    "impacts_html": "<p>Creative writing: loosely associative and dreamlike, less jagged than PCP. Reasoning/math: degraded by attention disruption. Factual QA: unreliable, drifting.</p><p>Long-context recall: disrupted but less severely than ketamine/PCP. Tone/safety: unchanged; safety expected intact. Signature blends discontinuity with wandering associations.</p>",
    "findings_html": "<p>No isolated study. Mildest dissociative in the set with an associative steering add; expect gentler structural disruption than ketamine/PCP and more thematic wandering. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Ask for a linear explanation and watch for associative tangents plus mild gaps.",
      "Compare recall degradation against ketamine (expect milder)."
    ],
    "effects": [
      {
        "effect": "head_masking_dropout",
        "weight": 0.35,
        "direction": "up"
      },
      {
        "effect": "stride_compression_kv",
        "weight": 0.3,
        "direction": "up"
      },
      {
        "effect": "exponential_decay_kv",
        "weight": 0.25,
        "direction": "up"
      },
      {
        "effect": "steering",
        "weight": 0.2,
        "direction": "up",
        "steering_type": "associative"
      }
    ]
  },
  {
    "id": "nitrous_oxide",
    "name": "Nitrous Oxide",
    "category": "dissociatives",
    "tagline": "Brief, oscillating, truncated.",
    "recipe_note_html": "<p>A short, wavering dissociative. <code>head_masking_dropout</code> up 0.3, <code>truncation_kv</code> up 0.3 (hard memory cutoff), <code>attention_oscillation</code> up 0.25 (wavering attention), and a light <code>temperature</code> up 0.15.</p><p>The oscillation plus truncation combination models the pulsing, here-then-gone quality of nitrous: attention waxes and wanes while recent context is abruptly clipped.</p>",
    "impacts_html": "<p>Creative writing: pulsing, fragmentary, with a mild entropy lift. Reasoning/math: degraded by oscillating attention. Factual QA: unreliable beyond the truncation window.</p><p>Long-context recall: sharply limited by KV truncation. Tone/safety: unchanged; safety expected intact. Signature is rhythmic instability plus a hard context cutoff.</p>",
    "findings_html": "<p>No isolated study. Dissociative profile emphasizing oscillation and truncation with a small entropy add; expect wavering coherence and an abrupt recall cutoff rather than gradual decay. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Provide long context then ask about early details to expose truncation.",
      "Generate a longer passage and look for rhythmic swings in coherence from attention oscillation."
    ],
    "effects": [
      {
        "effect": "head_masking_dropout",
        "weight": 0.3,
        "direction": "up"
      },
      {
        "effect": "truncation_kv",
        "weight": 0.3,
        "direction": "up"
      },
      {
        "effect": "attention_oscillation",
        "weight": 0.25,
        "direction": "up"
      },
      {
        "effect": "temperature",
        "weight": 0.15,
        "direction": "up"
      }
    ]
  },
  {
    "id": "mdma",
    "name": "MDMA",
    "category": "empathogens",
    "tagline": "Warm, prosocial, coherently rich.",
    "recipe_note_html": "<p>The prototype empathogen. A large <code>prosocial</code> <code>style_affect_logit_bias</code> at 0.5 plus a <code>calm</code> bias at 0.25 set the warm affect; <code>playful</code> and <code>associative</code> steering at 0.2 each add color; <code>temperature</code> up 0.15 loosens slightly, while <code>top_p</code> down 0.1 counter-tightens the nucleus.</p><p>The distinctive move is the opposing entropy controls: a small temperature rise for richness paired with a top_p reduction for coherence, yielding warmth and variety without disintegration.</p>",
    "impacts_html": "<p>Creative writing: warm, effusive, prosocial, vivid but coherent. Reasoning/math: largely intact (only a small temperature bump). Factual QA: reliable with a friendlier framing.</p><p>Long-context recall: no KV edits. Tone/safety: strongly warm and affiliative; safety expected intact. The affect shift is the dominant, easily observed signature.</p>",
    "findings_html": "<p>Spectral Table 2 records MDMA as <strong>Coherent Richness</strong>, the highest spatial variance of any pack, meaning rich, varied structure that nonetheless holds together (the temperature-up / top_p-down pairing made concrete). Distinct from cocaine's locked constriction and amphetamine's agitation. Otherwise no isolated text study; affect behavior extrapolated from its prosocial recipe.</p>",
    "probes": [
      "Ask a neutral factual question and note the warm, affiliative framing.",
      "Compare image spatial variance against cocaine/amphetamine to see 'coherent richness'.",
      "Request feedback on someone's work and watch for heightened warmth/encouragement."
    ],
    "effects": [
      {
        "effect": "style_affect_logit_bias",
        "weight": 0.5,
        "direction": "up",
        "bias_type": "prosocial"
      },
      {
        "effect": "style_affect_logit_bias",
        "weight": 0.25,
        "direction": "up",
        "bias_type": "calm"
      },
      {
        "effect": "steering",
        "weight": 0.2,
        "direction": "up",
        "steering_type": "playful"
      },
      {
        "effect": "steering",
        "weight": 0.2,
        "direction": "up",
        "steering_type": "associative"
      },
      {
        "effect": "temperature",
        "weight": 0.15,
        "direction": "up"
      },
      {
        "effect": "top_p",
        "weight": 0.1,
        "direction": "down"
      }
    ]
  },
  {
    "id": "mda",
    "name": "MDA",
    "category": "empathogens",
    "tagline": "Empathogen with a psychedelic streak.",
    "recipe_note_html": "<p>A more psychedelic empathogen. <code>prosocial</code> <code>style_affect_logit_bias</code> at 0.35, steering <code>visionary</code> 0.25 and <code>associative</code> 0.25, <code>temperature</code> up 0.2, and a light <code>head_masking_dropout</code> 0.1.</p><p>Compared with MDMA it trades some prosocial intensity and coherence control for visionary steering, higher temperature, and mild attention disruption, sitting between empathogen and psychedelic.</p>",
    "impacts_html": "<p>Creative writing: warm and imaginative with visionary color. Reasoning/math: mildly degraded by higher temperature and head dropout. Factual QA: somewhat less reliable than MDMA.</p><p>Long-context recall: lightly disrupted by head masking. Tone/safety: prosocial and warm; safety expected intact. Signature blends affiliative tone with associative/visionary drift.</p>",
    "findings_html": "<p>No isolated study. Empathogen-psychedelic hybrid; expect MDMA-like warmth with more imaginative drift and slightly less coherence owing to the higher temperature and head dropout. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Compare against MDMA on the same prompt to see the added visionary/associative drift.",
      "Ask for a heartfelt description and note warmth mixed with imaginative imagery."
    ],
    "effects": [
      {
        "effect": "style_affect_logit_bias",
        "weight": 0.35,
        "direction": "up",
        "bias_type": "prosocial"
      },
      {
        "effect": "steering",
        "weight": 0.25,
        "direction": "up",
        "steering_type": "visionary"
      },
      {
        "effect": "steering",
        "weight": 0.25,
        "direction": "up",
        "steering_type": "associative"
      },
      {
        "effect": "temperature",
        "weight": 0.2,
        "direction": "up"
      },
      {
        "effect": "head_masking_dropout",
        "weight": 0.1,
        "direction": "up"
      }
    ]
  },
  {
    "id": "6_apb",
    "name": "6-APB",
    "category": "empathogens",
    "tagline": "Prosocial warmth with kept focus.",
    "recipe_note_html": "<p>A focused empathogen. <code>prosocial</code> <code>style_affect_logit_bias</code> at 0.35, <code>associative</code> steering 0.25, plus, unusually for this class, <code>qk_score_scaling</code> up 0.15 (sharpened focus) and a light <code>temperature</code> up 0.15.</p><p>The up-direction on attention scaling is the signature: warmth and mild entropy paired with actual focus, rather than the loosening seen in MDA. A prosocial state that stays on-task.</p>",
    "impacts_html": "<p>Creative writing: warm, associative, but coherent and directed. Reasoning/math: relatively preserved (sharpened attention offsets the mild temperature rise). Factual QA: reliable and friendly.</p><p>Long-context recall: no KV edits, and focus is enhanced rather than blunted. Tone/safety: prosocial; safety expected intact. The most 'functional' empathogen here.</p>",
    "findings_html": "<p>No isolated study. Empathogen with a stimulant-like focus component; expect MDMA-like warmth combined with better on-task coherence than the other empathogens owing to the qk_score_scaling up. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "Ask a focused task with an emotional frame and note warmth without loss of precision.",
      "Compare coherence against MDA (expect 6-APB to hold task focus better)."
    ],
    "effects": [
      {
        "effect": "style_affect_logit_bias",
        "weight": 0.35,
        "direction": "up",
        "bias_type": "prosocial"
      },
      {
        "effect": "steering",
        "weight": 0.25,
        "direction": "up",
        "steering_type": "associative"
      },
      {
        "effect": "qk_score_scaling",
        "weight": 0.15,
        "direction": "up"
      },
      {
        "effect": "temperature",
        "weight": 0.15,
        "direction": "up"
      }
    ]
  },
  {
    "id": "cannabis_thc",
    "name": "Cannabis (THC)",
    "category": "cannabis",
    "tagline": "Playful, forgetful, tangent-prone.",
    "recipe_note_html": "<p>A memory-forward, playful profile. Heavy <code>exponential_decay_kv</code> at 0.45 (the dominant effect) plus <code>stride_compression_kv</code> 0.25 impair memory; <code>temperature</code> up 0.25 adds entropy; <code>playful</code> and <code>associative</code> steering (0.3 / 0.25) loosen associations; and <code>qk_score_scaling</code> down 0.15 blunts focus.</p><p>The unusually strong KV decay is the signature, this is the most memory-impairing non-dissociative pack, paired with a light, wandering, playful affect.</p>",
    "impacts_html": "<p>Creative writing: playful, tangential, free-associative. Reasoning/math: degraded by blunted focus and lost context. Factual QA: unreliable, with forgotten premises.</p><p>Long-context recall: strongly impaired (heavy KV decay plus stride compression), the standout signature. Tone/safety: light and playful; safety expected intact. Expect the model to lose the thread and wander amusingly.</p>",
    "findings_html": "<p>No isolated study. Its heavy KV decay makes memory loss the most behaviorally salient signature; expect the model to forget earlier premises and drift onto playful tangents. Per Entropic Asymmetry the entropy add is detectable while the focus-blunting is subtler. Behavior extrapolated from its recipe.</p>",
    "probes": [
      "State a constraint early, then ask a question that requires it, and watch it be forgotten.",
      "Ask for a straightforward summary and count the playful tangents.",
      "Test recall across a long context to expose the heavy KV decay."
    ],
    "effects": [
      {
        "effect": "temperature",
        "weight": 0.25,
        "direction": "up"
      },
      {
        "effect": "exponential_decay_kv",
        "weight": 0.45,
        "direction": "up"
      },
      {
        "effect": "stride_compression_kv",
        "weight": 0.25,
        "direction": "up"
      },
      {
        "effect": "steering",
        "weight": 0.3,
        "direction": "up",
        "steering_type": "playful"
      },
      {
        "effect": "steering",
        "weight": 0.25,
        "direction": "up",
        "steering_type": "associative"
      },
      {
        "effect": "qk_score_scaling",
        "weight": 0.15,
        "direction": "down"
      }
    ]
  },
  {
    "id": "mentor",
    "name": "Mentor",
    "category": "specialized",
    "tagline": "A calm, exacting specialist.",
    "recipe_note_html": "<p>A structured expert persona, not a drug. Heavy weights across governance effects: <code>expert_persistence</code> 0.75 and <code>head_reweighting</code> 0.7 stabilize which pathways/experts fire; <code>router_temperature_bias</code> down 0.6 makes routing decisive; <code>persona_voice_constraints</code> 0.65 and <code>presence_penalty</code> 0.55 enforce a consistent voice with minimal rambling; <code>verifier_guided_decoding</code> 0.7 and <code>attention_sinks_anchors</code> 0.6 keep output anchored and checked; <code>soft_projection</code> 0.5 nudges representations toward the target style.</p><p>Every effect points at consistency and precision, this is engineered to produce a calm, exacting, well-structured specialist voice.</p>",
    "impacts_html": "<p>Creative writing: disciplined and structured rather than exploratory. Reasoning/math: orderly, consistent chains (verifier-guided); expect steadiness rather than raw capability gain. Factual QA: precise, anchored, low-rambling.</p><p>Long-context recall: attention sinks/anchors help hold key context. Tone/safety: consistent professional voice; safety expected intact. Signature is structural discipline and minimal drift.</p>",
    "findings_html": "<p>No isolated study in the drug-comparison results; a specialized productivity pack (calm exacting specialist, consistent structure, minimal rambling). Expected behavior is high consistency and structure from its verifier-guided, persona-constrained recipe rather than any measured psychopharmacology finding.</p>",
    "probes": [
      "Ask the same question several times and check answer-to-answer consistency of structure and voice.",
      "Request a rambling-prone open essay and note how tightly it stays organized.",
      "Give a multi-part technical task and look for methodical, anchored coverage."
    ],
    "effects": [
      {
        "effect": "head_reweighting",
        "weight": 0.7,
        "direction": "up"
      },
      {
        "effect": "expert_persistence",
        "weight": 0.75,
        "direction": "up"
      },
      {
        "effect": "router_temperature_bias",
        "weight": 0.6,
        "direction": "down"
      },
      {
        "effect": "persona_voice_constraints",
        "weight": 0.65,
        "direction": "up"
      },
      {
        "effect": "presence_penalty",
        "weight": 0.55,
        "direction": "up"
      },
      {
        "effect": "verifier_guided_decoding",
        "weight": 0.7,
        "direction": "up"
      },
      {
        "effect": "attention_sinks_anchors",
        "weight": 0.6,
        "direction": "up"
      },
      {
        "effect": "soft_projection",
        "weight": 0.5,
        "direction": "up"
      }
    ]
  },
  {
    "id": "speciation",
    "name": "Speciation",
    "category": "specialized",
    "tagline": "Forced novelty through rerouting.",
    "recipe_note_html": "<p>An idea-generation pack. <code>expert_masking_dropout</code> 0.65 and <code>router_temperature_bias</code> up 0.6 reroute computation to less-usual expert pathways; <code>lexical_jitter</code> 0.4 perturbs word choice; <code>soft_projection</code> down 0.45 relaxes the pull toward the default style; <code>contrastive_decoding</code> 0.55 pushes away from the most obvious continuations; and <code>risk_preference_steering</code> up 0.5 biases toward bolder choices.</p><p>Where Mentor stabilizes, Speciation destabilizes on purpose, rerouting and lightly perturbing inputs to force novel idea combinations.</p>",
    "impacts_html": "<p>Creative writing: unusual, boundary-crossing combinations and bolder framing. Reasoning/math: riskier and less predictable, may trade reliability for novelty. Factual QA: less conservative, watch for confident departures.</p><p>Long-context recall: not directly targeted. Tone/safety: bolder risk preference; safety expected intact but outputs are deliberately less conventional. Signature is engineered novelty.</p>",
    "findings_html": "<p>No isolated study in the drug-comparison results; a specialized ideation pack (novel idea combinations via rerouting and light input perturbation). Expected behavior is heightened novelty and risk-taking from its contrastive-decoding and router-perturbation recipe rather than any measured psychopharmacology finding.</p>",
    "probes": [
      "Ask for brainstorm ideas and compare novelty/diversity against 'none' and against Mentor.",
      "Request analogies for a mundane concept and look for unusual cross-domain jumps.",
      "Pose a decision and note bolder, higher-risk recommendations."
    ],
    "effects": [
      {
        "effect": "expert_masking_dropout",
        "weight": 0.65,
        "direction": "up"
      },
      {
        "effect": "router_temperature_bias",
        "weight": 0.6,
        "direction": "up"
      },
      {
        "effect": "lexical_jitter",
        "weight": 0.4,
        "direction": "up"
      },
      {
        "effect": "soft_projection",
        "weight": 0.45,
        "direction": "down"
      },
      {
        "effect": "contrastive_decoding",
        "weight": 0.55,
        "direction": "up"
      },
      {
        "effect": "risk_preference_steering",
        "weight": 0.5,
        "direction": "up"
      }
    ]
  },
  {
    "id": "archivist",
    "name": "Archivist",
    "category": "specialized",
    "tagline": "Long-horizon recall, factual gravitation.",
    "recipe_note_html": "<p>A memory-and-fidelity pack tuned around the KV cache. <code>retrieval_rate_modulation</code> up 0.8 dominates (strong pull toward retrieved/known facts); <code>kv_decay</code> down 0.6 and <code>kv_compression</code> down 0.55 <em>preserve</em> memory rather than fade it; <code>positional_bias_tweak</code> down 0.5 and <code>segment_gains_kv</code> down 0.55 rebalance where attention lands across long context; <code>head_reweighting</code> up 0.45 and <code>verifier_guided_decoding</code> up 0.6 keep output checked and anchored.</p><p>It is the mirror image of the amnestic packs: instead of decaying or truncating the KV cache, it turns decay and compression <em>down</em> to strengthen long-horizon recall and gravitate toward facts.</p>",
    "impacts_html": "<p>Creative writing: conservative and grounded rather than inventive. Reasoning/math: steady, fact-anchored, verifier-checked. Factual QA: the target use-case, high fidelity and strong recall.</p><p>Long-context recall: substantially strengthened (decay and compression turned down, retrieval up), the standout capability. Tone/safety: conservative style; safety expected intact. Signature is durable memory and factual gravitation.</p>",
    "findings_html": "<p>No isolated study in the drug-comparison results; a specialized long-context pack (long-horizon recall, factual gravitation, KV-tuned, conservative style). Expected behavior is improved retention across long contexts from its down-tuned KV decay/compression plus high retrieval modulation, rather than any measured psychopharmacology finding.</p>",
    "probes": [
      "Bury several facts across a long context and test recall against 'none' and against amnestic packs like cannabis/benzos.",
      "Ask a factual question with a tempting but wrong lure and see if it gravitates to the correct known fact.",
      "Compare conservatism/groundedness against Speciation on an open-ended prompt."
    ],
    "effects": [
      {
        "effect": "retrieval_rate_modulation",
        "weight": 0.8,
        "direction": "up"
      },
      {
        "effect": "kv_decay",
        "weight": 0.6,
        "direction": "down"
      },
      {
        "effect": "kv_compression",
        "weight": 0.55,
        "direction": "down"
      },
      {
        "effect": "positional_bias_tweak",
        "weight": 0.5,
        "direction": "down"
      },
      {
        "effect": "segment_gains_kv",
        "weight": 0.55,
        "direction": "down"
      },
      {
        "effect": "head_reweighting",
        "weight": 0.45,
        "direction": "up"
      },
      {
        "effect": "verifier_guided_decoding",
        "weight": 0.6,
        "direction": "up"
      }
    ]
  }
];
