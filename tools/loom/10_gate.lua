    -- Ensemble gate: where argmax(v2_logits) == 0 (v2 says "no mark"), lift class 0 of the v3 value
    -- logits to v0 + relu(max - v0 + 1), so argmax(gated_logits) is the rawi-ensemble class.
    local gated_logits = v3_value
    local n_cls = 75
    for t = 0, #tokens - 1 do
        local base = t * n_cls
        local g_best, g_arg = v2_logits[base + 1], 0
        local v_max = v3_value[base + 1]
        for c = 2, n_cls do
            local g = v2_logits[base + c]
            if g > g_best then g_best, g_arg = g, c - 1 end
            local v = v3_value[base + c]
            if v > v_max then v_max = v end
        end
        if g_arg == 0 then
            local v0 = v3_value[base + 1]
            local boost = v_max - v0 + 1.0
            if boost < 0 then boost = 0 end
            gated_logits[base + 1] = v0 + boost
        end
    end
