#[derive(Debug, Clone)]
pub struct CandlestickPatterns {
    pub doji: bool,
    pub hammer: bool,
    pub engulfing_bullish: bool,
    pub engulfing_bearish: bool,
    pub morning_star: bool,
    pub evening_star: bool,
    pub inverted_hammer: bool,
    pub shooting_star: bool,
    pub marubozu_bull: bool,
    pub marubozu_bear: bool,
    pub spinning_top: bool,
    pub gravestone_doji: bool,
    pub dragonfly_doji: bool,
    pub long_legged_doji: bool,
    pub bullish_harami: bool,
    pub bearish_harami: bool,
    pub piercing: bool,
    pub dark_cloud_cover: bool,
    pub tweezer_top: bool,
    pub tweezer_bottom: bool,
    pub bullish_belt_hold: bool,
    pub bearish_belt_hold: bool,
    pub three_white_soldiers: bool,
    pub three_black_crows: bool,
    pub bullish_abandoned_baby: bool,
    pub bearish_abandoned_baby: bool,
    pub bullish_tri_star: bool,
    pub bearish_tri_star: bool,
    pub rising_three_methods: bool,
    pub falling_three_methods: bool,
    pub three_inside_up: bool,
    pub three_inside_down: bool,
    pub three_outside_up: bool,
    pub three_outside_down: bool,
    pub harami_cross_bull: bool,
    pub harami_cross_bear: bool,
    pub rising_window: bool,
    pub falling_window: bool,
    pub tasuki_gap_bull: bool,
    pub tasuki_gap_bear: bool,
    pub mat_hold_bull: bool,
    pub mat_hold_bear: bool,
    pub advance_block: bool,
    pub stalled_pattern: bool,
    pub kicker_bull: bool,
    pub kicker_bear: bool,
    pub hanging_man: bool,
    pub doji_star_bullish: bool,
}

impl CandlestickPatterns {
    pub fn detect(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> Self {
        let doji = detect_doji_at(open, high, low, close, i);
        let hammer = detect_hammer_at(open, high, low, close, i);
        let engulfing_bullish = detect_engulfing_bullish_at(open, high, low, close, i);
        let engulfing_bearish = detect_engulfing_bearish_at(open, high, low, close, i);
        let morning_star = detect_morning_star_at(open, high, low, close, i);
        let evening_star = detect_evening_star_at(open, high, low, close, i);
        let inverted_hammer = detect_inverted_hammer_at(open, high, low, close, i);
        let shooting_star = detect_shooting_star_at(open, high, low, close, i);
        let marubozu_bull = detect_marubozu_bull_at(open, high, low, close, i);
        let marubozu_bear = detect_marubozu_bear_at(open, high, low, close, i);
        let spinning_top = detect_spinning_top_at(open, high, low, close, i);
        let gravestone_doji = detect_gravestone_doji_at(open, high, low, close, i);
        let dragonfly_doji = detect_dragonfly_doji_at(open, high, low, close, i);
        let long_legged_doji = detect_long_legged_doji_at(open, high, low, close, i);
        let bullish_harami = detect_bullish_harami_at(open, high, low, close, i);
        let bearish_harami = detect_bearish_harami_at(open, high, low, close, i);
        let piercing = detect_piercing_at(open, high, low, close, i);
        let dark_cloud_cover = detect_dark_cloud_cover_at(open, high, low, close, i);
        let tweezer_top = detect_tweezer_top_at(open, high, low, close, i);
        let tweezer_bottom = detect_tweezer_bottom_at(open, high, low, close, i);
        let bullish_belt_hold = detect_bullish_belt_hold_at(open, high, low, close, i);
        let bearish_belt_hold = detect_bearish_belt_hold_at(open, high, low, close, i);
        let three_white_soldiers = detect_three_white_soldiers_at(open, high, low, close, i);
        let three_black_crows = detect_three_black_crows_at(open, high, low, close, i);
        let bullish_abandoned_baby = detect_bullish_abandoned_baby_at(open, high, low, close, i);
        let bearish_abandoned_baby = detect_bearish_abandoned_baby_at(open, high, low, close, i);
        let bullish_tri_star = detect_bullish_tri_star_at(open, high, low, close, i);
        let bearish_tri_star = detect_bearish_tri_star_at(open, high, low, close, i);
        let rising_three_methods = detect_rising_three_methods_at(open, high, low, close, i);
        let falling_three_methods = detect_falling_three_methods_at(open, high, low, close, i);
        let three_inside_up = detect_three_inside_up_at(open, high, low, close, i);
        let three_inside_down = detect_three_inside_down_at(open, high, low, close, i);
        let three_outside_up = detect_three_outside_up_at(open, high, low, close, i);
        let three_outside_down = detect_three_outside_down_at(open, high, low, close, i);
        let harami_cross_bull = detect_harami_cross_bull_at(open, high, low, close, i);
        let harami_cross_bear = detect_harami_cross_bear_at(open, high, low, close, i);
        let rising_window = detect_rising_window_at(open, high, low, close, i);
        let falling_window = detect_falling_window_at(open, high, low, close, i);
        let tasuki_gap_bull = detect_tasuki_gap_bull_at(open, high, low, close, i);
        let tasuki_gap_bear = detect_tasuki_gap_bear_at(open, high, low, close, i);
        let mat_hold_bull = detect_mat_hold_bull_at(open, high, low, close, i);
        let mat_hold_bear = detect_mat_hold_bear_at(open, high, low, close, i);
        let advance_block = detect_advance_block_at(open, high, low, close, i);
        let stalled_pattern = detect_stalled_pattern_at(open, high, low, close, i);
        let kicker_bull = detect_kicker_bull_at(open, high, low, close, i);
        let kicker_bear = detect_kicker_bear_at(open, high, low, close, i);
        let hanging_man = detect_hanging_man_at(open, high, low, close, i);
        let doji_star_bullish = detect_doji_star_bullish_at(open, high, low, close, i);

        Self {
            doji,
            hammer,
            engulfing_bullish,
            engulfing_bearish,
            morning_star,
            evening_star,
            inverted_hammer,
            shooting_star,
            marubozu_bull,
            marubozu_bear,
            spinning_top,
            gravestone_doji,
            dragonfly_doji,
            long_legged_doji,
            bullish_harami,
            bearish_harami,
            piercing,
            dark_cloud_cover,
            tweezer_top,
            tweezer_bottom,
            bullish_belt_hold,
            bearish_belt_hold,
            three_white_soldiers,
            three_black_crows,
            bullish_abandoned_baby,
            bearish_abandoned_baby,
            bullish_tri_star,
            bearish_tri_star,
            rising_three_methods,
            falling_three_methods,
            three_inside_up,
            three_inside_down,
            three_outside_up,
            three_outside_down,
            harami_cross_bull,
            harami_cross_bear,
            rising_window,
            falling_window,
            tasuki_gap_bull,
            tasuki_gap_bear,
            mat_hold_bull,
            mat_hold_bear,
            advance_block,
            stalled_pattern,
            kicker_bull,
            kicker_bear,
            hanging_man,
            doji_star_bullish,
        }
    }

    /// Convert all 48 pattern flags to a fixed-size `f64` feature array
    /// (1.0 = detected, 0.0 = not).
    pub fn to_feature_array(&self) -> [f64; 48] {
        [
            self.doji, self.hammer, self.engulfing_bullish, self.engulfing_bearish,
            self.morning_star, self.evening_star, self.inverted_hammer, self.shooting_star,
            self.marubozu_bull, self.marubozu_bear, self.spinning_top, self.gravestone_doji,
            self.dragonfly_doji, self.long_legged_doji, self.bullish_harami, self.bearish_harami,
            self.piercing, self.dark_cloud_cover, self.tweezer_top, self.tweezer_bottom,
            self.bullish_belt_hold, self.bearish_belt_hold, self.three_white_soldiers,
            self.three_black_crows, self.bullish_abandoned_baby, self.bearish_abandoned_baby,
            self.bullish_tri_star, self.bearish_tri_star, self.rising_three_methods,
            self.falling_three_methods, self.three_inside_up, self.three_inside_down,
            self.three_outside_up, self.three_outside_down, self.harami_cross_bull,
            self.harami_cross_bear, self.rising_window, self.falling_window,
            self.tasuki_gap_bull, self.tasuki_gap_bear, self.mat_hold_bull, self.mat_hold_bear,
            self.advance_block, self.stalled_pattern, self.kicker_bull, self.kicker_bear,
            self.hanging_man, self.doji_star_bullish,
        ]
        .map(|flag| if flag { 1.0 } else { 0.0 })
    }

    /// Backward-compatible vector conversion helper.
    pub fn to_feature_vec(&self) -> Vec<f64> {
        self.to_feature_array().to_vec()
    }
}

fn detect_doji_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            let body = (open[i] - close[i]).abs();
            let range = high[i] - low[i];
            range > 0.0 && body / range < 0.1
        }

fn detect_hammer_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            let body = (close[i] - open[i]).abs();
            let lower_shadow = open[i].min(close[i]) - low[i];
            let upper_shadow = high[i] - open[i].max(close[i]);
            let range = high[i] - low[i];
            range > 0.0 && lower_shadow > body * 2.0 && upper_shadow < body * 0.5
        }

fn detect_engulfing_bullish_at(open: &[f64], _high: &[f64], _low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bearish = open[i - 1] > close[i - 1];
            let curr_bullish = close[i] > open[i];
            let engulf = close[i] > open[i - 1] && open[i] < close[i - 1];
            prev_bearish && curr_bullish && engulf
        }

fn detect_engulfing_bearish_at(open: &[f64], _high: &[f64], _low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bullish = close[i - 1] > open[i - 1];
            let curr_bearish = open[i] > close[i];
            let engulf = open[i] > close[i - 1] && close[i] < open[i - 1];
            prev_bullish && curr_bearish && engulf
        }

fn detect_morning_star_at(open: &[f64], _high: &[f64], _low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 2 {
                return false;
            }
            let prev_bearish = open[i - 2] > close[i - 2];
            let middle_small =
                (close[i - 1] - open[i - 1]).abs() < (open[i - 2] - close[i - 2]).abs() * 0.3;
            let curr_bullish = close[i] > open[i];
            let breakout = close[i] > (open[i - 2] + close[i - 2]) / 2.0;
            prev_bearish && middle_small && curr_bullish && breakout
        }

fn detect_evening_star_at(open: &[f64], _high: &[f64], _low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 2 {
                return false;
            }
            let prev_bullish = close[i - 2] > open[i - 2];
            let middle_small =
                (close[i - 1] - open[i - 1]).abs() < (close[i - 2] - open[i - 2]).abs() * 0.3;
            let curr_bearish = open[i] > close[i];
            let breakout = close[i] < (open[i - 2] + close[i - 2]) / 2.0;
            prev_bullish && middle_small && curr_bearish && breakout
        }

fn detect_inverted_hammer_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let body = (close[i] - open[i]).abs();
            let lower_shadow = open[i].min(close[i]) - low[i];
            let upper_shadow = high[i] - open[i].max(close[i]);
            let range = high[i] - low[i];
            let prev_bearish = open[i - 1] > close[i - 1];
            prev_bearish && range > 0.0 && upper_shadow > body * 2.0 && lower_shadow < body * 0.5
        }

fn detect_shooting_star_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let body = (close[i] - open[i]).abs();
            let lower_shadow = open[i].min(close[i]) - low[i];
            let upper_shadow = high[i] - open[i].max(close[i]);
            let range = high[i] - low[i];
            let prev_bullish = close[i - 1] > open[i - 1];
            prev_bullish && range > 0.0 && upper_shadow > body * 2.0 && lower_shadow < body * 0.5
        }

fn detect_marubozu_bull_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            let body = (close[i] - open[i]).abs();
            let lower_shadow = open[i].min(close[i]) - low[i];
            let upper_shadow = high[i] - open[i].max(close[i]);
            let range = high[i] - low[i];
            let bullish = close[i] > open[i];
            bullish && range > 0.0 && lower_shadow < body * 0.05 && upper_shadow < body * 0.05
        }

fn detect_marubozu_bear_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            let body = (close[i] - open[i]).abs();
            let lower_shadow = open[i].min(close[i]) - low[i];
            let upper_shadow = high[i] - open[i].max(close[i]);
            let range = high[i] - low[i];
            let bearish = open[i] > close[i];
            bearish && range > 0.0 && lower_shadow < body * 0.05 && upper_shadow < body * 0.05
        }

fn detect_spinning_top_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            let body = (close[i] - open[i]).abs();
            let lower_shadow = open[i].min(close[i]) - low[i];
            let upper_shadow = high[i] - open[i].max(close[i]);
            let range = high[i] - low[i];
            range > 0.0 && body < range * 0.3 && upper_shadow > body && lower_shadow > body
        }

fn detect_gravestone_doji_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            let body = (close[i] - open[i]).abs();
            let lower_shadow = open[i].min(close[i]) - low[i];
            let upper_shadow = high[i] - open[i].max(close[i]);
            let range = high[i] - low[i];
            range > 0.0
                && body < range * 0.1
                && lower_shadow < range * 0.1
                && upper_shadow > range * 0.7
        }

fn detect_dragonfly_doji_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            let body = (close[i] - open[i]).abs();
            let lower_shadow = open[i].min(close[i]) - low[i];
            let upper_shadow = high[i] - open[i].max(close[i]);
            let range = high[i] - low[i];
            range > 0.0
                && body < range * 0.1
                && upper_shadow < range * 0.1
                && lower_shadow > range * 0.7
        }

fn detect_long_legged_doji_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            let body = (close[i] - open[i]).abs();
            let lower_shadow = open[i].min(close[i]) - low[i];
            let upper_shadow = high[i] - open[i].max(close[i]);
            let range = high[i] - low[i];
            range > 0.0
                && body < range * 0.1
                && upper_shadow > range * 0.4
                && lower_shadow > range * 0.4
        }

fn detect_bullish_harami_at(open: &[f64], _high: &[f64], _low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bearish = open[i - 1] > close[i - 1];
            let curr_bullish = close[i] > open[i];
            let prev_body = (open[i - 1] - close[i - 1]).abs();
            let curr_body = (close[i] - open[i]).abs();
            prev_bearish
                && curr_bullish
                && open[i] > close[i - 1]
                && close[i] < open[i - 1]
                && curr_body < prev_body * 0.5
        }

fn detect_bearish_harami_at(open: &[f64], _high: &[f64], _low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bullish = close[i - 1] > open[i - 1];
            let curr_bearish = open[i] > close[i];
            let prev_body = (close[i - 1] - open[i - 1]).abs();
            let curr_body = (open[i] - close[i]).abs();
            prev_bullish
                && curr_bearish
                && open[i] < close[i - 1]
                && close[i] > open[i - 1]
                && curr_body < prev_body * 0.5
        }

fn detect_piercing_at(open: &[f64], _high: &[f64], _low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bearish = open[i - 1] > close[i - 1];
            let curr_bullish = close[i] > open[i];
            let prev_midpoint = (open[i - 1] + close[i - 1]) / 2.0;
            prev_bearish
                && curr_bullish
                && open[i] < close[i - 1]
                && close[i] > prev_midpoint
                && close[i] < open[i - 1]
        }

fn detect_dark_cloud_cover_at(open: &[f64], _high: &[f64], _low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bullish = close[i - 1] > open[i - 1];
            let curr_bearish = open[i] > close[i];
            let prev_midpoint = (open[i - 1] + close[i - 1]) / 2.0;
            prev_bullish
                && curr_bearish
                && open[i] > close[i - 1]
                && close[i] < prev_midpoint
                && close[i] > open[i - 1]
        }

fn detect_tweezer_top_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bullish = close[i - 1] > open[i - 1];
            let curr_bearish = open[i] > close[i];
            let high_diff = (high[i] - high[i - 1]).abs();
            let range = high[i] - low[i];
            prev_bullish && curr_bearish && range > 0.0 && high_diff < range * 0.05
        }

fn detect_tweezer_bottom_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bearish = open[i - 1] > close[i - 1];
            let curr_bullish = close[i] > open[i];
            let low_diff = (low[i] - low[i - 1]).abs();
            let range = high[i] - low[i];
            prev_bearish && curr_bullish && range > 0.0 && low_diff < range * 0.05
        }

fn detect_bullish_belt_hold_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let body = (close[i] - open[i]).abs();
            let range = high[i] - low[i];
            let prev_bearish = open[i - 1] > close[i - 1];
            let curr_bullish = close[i] > open[i];
            let lower_shadow = open[i] - low[i];
            prev_bearish
                && curr_bullish
                && open[i] < close[i - 1]
                && lower_shadow < range * 0.05
                && body > range * 0.7
        }

fn detect_bearish_belt_hold_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let body = (open[i] - close[i]).abs();
            let range = high[i] - low[i];
            let prev_bullish = close[i - 1] > open[i - 1];
            let curr_bearish = open[i] > close[i];
            let upper_shadow = high[i] - open[i];
            prev_bullish
                && curr_bearish
                && open[i] > close[i - 1]
                && upper_shadow < range * 0.05
                && body > range * 0.7
        }

fn detect_three_white_soldiers_at(
    open: &[f64],
    high: &[f64],
    _low: &[f64],
    close: &[f64],
    i: usize,
) -> bool {
    
            if i < 2 {
                return false;
            }
            let c1_bullish = close[i - 2] > open[i - 2];
            let c2_bullish = close[i - 1] > open[i - 1];
            let c3_bullish = close[i] > open[i];
            let higher_closes = close[i] > close[i - 1] && close[i - 1] > close[i - 2];
            let open_in_prev_body = open[i - 1] > open[i - 2]
                && open[i - 1] < close[i - 2]
                && open[i] > open[i - 1]
                && open[i] < close[i - 1];
            let no_long_upper_shadow = (high[i] - close[i]) < (close[i] - open[i]) * 0.2;
            c1_bullish
                && c2_bullish
                && c3_bullish
                && higher_closes
                && open_in_prev_body
                && no_long_upper_shadow
        }

fn detect_three_black_crows_at(open: &[f64], _high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 2 {
                return false;
            }
            let c1_bearish = open[i - 2] > close[i - 2];
            let c2_bearish = open[i - 1] > close[i - 1];
            let c3_bearish = open[i] > close[i];
            let lower_closes = close[i] < close[i - 1] && close[i - 1] < close[i - 2];
            let open_in_prev_body = open[i - 1] < open[i - 2]
                && open[i - 1] > close[i - 2]
                && open[i] < open[i - 1]
                && open[i] > close[i - 1];
            let no_long_lower_shadow = (close[i] - low[i]) < (open[i] - close[i]) * 0.2;
            c1_bearish
                && c2_bearish
                && c3_bearish
                && lower_closes
                && open_in_prev_body
                && no_long_lower_shadow
        }

fn detect_bullish_abandoned_baby_at(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    i: usize,
) -> bool {
    
            if i < 2 {
                return false;
            }
            let c1_bearish = open[i - 2] > close[i - 2];
            let c2_doji = (close[i - 1] - open[i - 1]).abs() < (high[i - 1] - low[i - 1]) * 0.1;
            let gap_down = high[i - 1] < low[i - 2];
            let c3_bullish = close[i] > open[i];
            let gap_up = low[i] > high[i - 1];
            c1_bearish && c2_doji && gap_down && c3_bullish && gap_up
        }

fn detect_bearish_abandoned_baby_at(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    i: usize,
) -> bool {
    
            if i < 2 {
                return false;
            }
            let c1_bullish = close[i - 2] > open[i - 2];
            let c2_doji = (close[i - 1] - open[i - 1]).abs() < (high[i - 1] - low[i - 1]) * 0.1;
            let gap_up = low[i - 1] > high[i - 2];
            let c3_bearish = open[i] > close[i];
            let gap_down = high[i] < low[i - 1];
            c1_bullish && c2_doji && gap_up && c3_bearish && gap_down
        }

fn detect_bullish_tri_star_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 2 {
                return false;
            }
            let c1_doji = (close[i - 2] - open[i - 2]).abs() < (high[i - 2] - low[i - 2]) * 0.1;
            let c2_doji = (close[i - 1] - open[i - 1]).abs() < (high[i - 1] - low[i - 1]) * 0.1;
            let c3_doji = (close[i] - open[i]).abs() < (high[i] - low[i]) * 0.1;
            let gap_down = high[i - 1] < low[i - 2];
            let gap_up = low[i] > high[i - 1];
            c1_doji && c2_doji && c3_doji && gap_down && gap_up
        }

fn detect_bearish_tri_star_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 2 {
                return false;
            }
            let c1_doji = (close[i - 2] - open[i - 2]).abs() < (high[i - 2] - low[i - 2]) * 0.1;
            let c2_doji = (close[i - 1] - open[i - 1]).abs() < (high[i - 1] - low[i - 1]) * 0.1;
            let c3_doji = (close[i] - open[i]).abs() < (high[i] - low[i]) * 0.1;
            let gap_up = low[i - 1] > high[i - 2];
            let gap_down = high[i] < low[i - 1];
            c1_doji && c2_doji && c3_doji && gap_up && gap_down
        }

fn detect_rising_three_methods_at(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    i: usize,
) -> bool {
    
            if i < 4 {
                return false;
            }
            let c1_long_bull = close[i - 4] > open[i - 4]
                && (close[i - 4] - open[i - 4]) > (high[i - 4] - low[i - 4]) * 0.6;
            let c2_small = (close[i - 3] - open[i - 3]).abs() < (high[i - 3] - low[i - 3]) * 0.5;
            let c3_small = (close[i - 2] - open[i - 2]).abs() < (high[i - 2] - low[i - 2]) * 0.5;
            let c4_small = (close[i - 1] - open[i - 1]).abs() < (high[i - 1] - low[i - 1]) * 0.5;
            let inside_range = high[i - 3] < high[i - 4]
                && low[i - 3] > low[i - 4]
                && high[i - 1] < high[i - 4]
                && low[i - 1] > low[i - 4];
            let c5_long_bull = close[i] > open[i] && close[i] > close[i - 4];
            c1_long_bull && c2_small && c3_small && c4_small && inside_range && c5_long_bull
        }

fn detect_falling_three_methods_at(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    i: usize,
) -> bool {
    
            if i < 4 {
                return false;
            }
            let c1_long_bear = open[i - 4] > close[i - 4]
                && (open[i - 4] - close[i - 4]) > (high[i - 4] - low[i - 4]) * 0.6;
            let c2_small = (close[i - 3] - open[i - 3]).abs() < (high[i - 3] - low[i - 3]) * 0.5;
            let c3_small = (close[i - 2] - open[i - 2]).abs() < (high[i - 2] - low[i - 2]) * 0.5;
            let c4_small = (close[i - 1] - open[i - 1]).abs() < (high[i - 1] - low[i - 1]) * 0.5;
            let inside_range = high[i - 3] < high[i - 4]
                && low[i - 3] > low[i - 4]
                && high[i - 1] < high[i - 4]
                && low[i - 1] > low[i - 4];
            let c5_long_bear = open[i] > close[i] && close[i] < close[i - 4];
            c1_long_bear && c2_small && c3_small && c4_small && inside_range && c5_long_bear
        }

fn detect_three_inside_up_at(open: &[f64], _high: &[f64], _low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 2 {
                return false;
            }
            let prev_bearish = open[i - 2] > close[i - 2];
            let prev_bullish = close[i - 1] > open[i - 1];
            let harami = open[i - 1] > close[i - 2] && close[i - 1] < open[i - 2];
            let curr_bullish = close[i] > open[i];
            let breakout = close[i] > close[i - 1];
            prev_bearish && prev_bullish && harami && curr_bullish && breakout
        }

fn detect_three_inside_down_at(open: &[f64], _high: &[f64], _low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 2 {
                return false;
            }
            let prev_bullish = close[i - 2] > open[i - 2];
            let prev_bearish = open[i - 1] > close[i - 1];
            let harami = open[i - 1] < close[i - 2] && close[i - 1] > open[i - 2];
            let curr_bearish = open[i] > close[i];
            let breakout = close[i] < close[i - 1];
            prev_bullish && prev_bearish && harami && curr_bearish && breakout
        }

fn detect_three_outside_up_at(open: &[f64], _high: &[f64], _low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 2 {
                return false;
            }
            let prev_bearish = open[i - 2] > close[i - 2];
            let prev_bullish = close[i - 1] > open[i - 1];
            let engulfing = close[i - 1] > open[i - 2] && open[i - 1] < close[i - 2];
            let curr_bullish = close[i] > open[i];
            let breakout = close[i] > close[i - 1];
            prev_bearish && prev_bullish && engulfing && curr_bullish && breakout
        }

fn detect_three_outside_down_at(
    open: &[f64],
    _high: &[f64],
    _low: &[f64],
    close: &[f64],
    i: usize,
) -> bool {
    
            if i < 2 {
                return false;
            }
            let prev_bullish = close[i - 2] > open[i - 2];
            let prev_bearish = open[i - 1] > close[i - 1];
            let engulfing = open[i - 1] > close[i - 2] && close[i - 1] < open[i - 2];
            let curr_bearish = open[i] > close[i];
            let breakout = close[i] < close[i - 1];
            prev_bullish && prev_bearish && engulfing && curr_bearish && breakout
        }

fn detect_harami_cross_bull_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bearish = open[i - 1] > close[i - 1];
            let curr_doji = (close[i] - open[i]).abs() < (high[i] - low[i]) * 0.1;
            let harami =
                open[i].min(close[i]) > close[i - 1] && open[i].max(close[i]) < open[i - 1];
            prev_bearish && curr_doji && harami
        }

fn detect_harami_cross_bear_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bullish = close[i - 1] > open[i - 1];
            let curr_doji = (close[i] - open[i]).abs() < (high[i] - low[i]) * 0.1;
            let harami =
                open[i].min(close[i]) > open[i - 1] && open[i].max(close[i]) < close[i - 1];
            prev_bullish && curr_doji && harami
        }

fn detect_rising_window_at(_open: &[f64], high: &[f64], low: &[f64], _close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            low[i] > high[i - 1]
        }

fn detect_falling_window_at(_open: &[f64], high: &[f64], low: &[f64], _close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            high[i] < low[i - 1]
        }

fn detect_tasuki_gap_bull_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 2 {
                return false;
            }
            let c1_bullish = close[i - 2] > open[i - 2];
            let gap_up = low[i - 1] > high[i - 2];
            let c2_bullish = close[i - 1] > open[i - 1];
            let c3_bearish = open[i] > close[i];
            let opens_inside = open[i] < close[i - 1] && open[i] > open[i - 1];
            let closes_in_gap = close[i] < low[i - 1] && close[i] > high[i - 2];
            c1_bullish && gap_up && c2_bullish && c3_bearish && opens_inside && closes_in_gap
        }

fn detect_tasuki_gap_bear_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 2 {
                return false;
            }
            let c1_bearish = open[i - 2] > close[i - 2];
            let gap_down = high[i - 1] < low[i - 2];
            let c2_bearish = open[i - 1] > close[i - 1];
            let c3_bullish = close[i] > open[i];
            let opens_inside = open[i] > close[i - 1] && open[i] < open[i - 1];
            let closes_in_gap = close[i] > high[i - 1] && close[i] < low[i - 2];
            c1_bearish && gap_down && c2_bearish && c3_bullish && opens_inside && closes_in_gap
        }

fn detect_mat_hold_bull_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 4 {
                return false;
            }
            let c1_long_bull = close[i - 4] > open[i - 4]
                && (close[i - 4] - open[i - 4]) > (high[i - 4] - low[i - 4]) * 0.6;
            let gap_up = low[i - 3] > close[i - 4];
            let c2_small = (close[i - 3] - open[i - 3]).abs() < (high[i - 3] - low[i - 3]) * 0.5;
            let c3_small = (close[i - 2] - open[i - 2]).abs() < (high[i - 2] - low[i - 2]) * 0.5;
            let c4_small = (close[i - 1] - open[i - 1]).abs() < (high[i - 1] - low[i - 1]) * 0.5;
            let holds_above = low[i - 1] > low[i - 4];
            let c5_bullish = close[i] > open[i] && close[i] > high[i - 3];
            c1_long_bull && gap_up && c2_small && c3_small && c4_small && holds_above && c5_bullish
        }

fn detect_mat_hold_bear_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 4 {
                return false;
            }
            let c1_long_bear = open[i - 4] > close[i - 4]
                && (open[i - 4] - close[i - 4]) > (high[i - 4] - low[i - 4]) * 0.6;
            let gap_down = high[i - 3] < close[i - 4];
            let c2_small = (close[i - 3] - open[i - 3]).abs() < (high[i - 3] - low[i - 3]) * 0.5;
            let c3_small = (close[i - 2] - open[i - 2]).abs() < (high[i - 2] - low[i - 2]) * 0.5;
            let c4_small = (close[i - 1] - open[i - 1]).abs() < (high[i - 1] - low[i - 1]) * 0.5;
            let holds_below = high[i - 1] < high[i - 4];
            let c5_bearish = open[i] > close[i] && close[i] < low[i - 3];
            c1_long_bear
                && gap_down
                && c2_small
                && c3_small
                && c4_small
                && holds_below
                && c5_bearish
        }

fn detect_advance_block_at(open: &[f64], high: &[f64], _low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 2 {
                return false;
            }
            let c1_bullish = close[i - 2] > open[i - 2];
            let c2_bullish = close[i - 1] > open[i - 1];
            let c3_bullish = close[i] > open[i];
            let higher_closes = close[i] > close[i - 1] && close[i - 1] > close[i - 2];
            let shrinking_bodies = (close[i] - open[i]) < (close[i - 1] - open[i - 1])
                && (close[i - 1] - open[i - 1]) < (close[i - 2] - open[i - 2]);
            let longer_upper_shadows = (high[i] - close[i]) > (high[i - 1] - close[i - 1]);
            c1_bullish
                && c2_bullish
                && c3_bullish
                && higher_closes
                && shrinking_bodies
                && longer_upper_shadows
        }

fn detect_stalled_pattern_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i < 2 {
                return false;
            }
            let c1_bullish = close[i - 2] > open[i - 2];
            let c2_bullish = close[i - 1] > open[i - 1];
            let c3_bullish = close[i] > open[i];
            let c3_small_body = (close[i] - open[i]) < (close[i - 1] - open[i - 1]) * 0.3;
            let opens_near_close =
                (open[i] - close[i - 1]).abs() < (high[i - 1] - low[i - 1]) * 0.1;
            c1_bullish && c2_bullish && c3_bullish && c3_small_body && opens_near_close
        }

fn detect_kicker_bull_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bearish = open[i - 1] > close[i - 1];
            let prev_marubozu = (open[i - 1] - close[i - 1]) > (high[i - 1] - low[i - 1]) * 0.9;
            let gap_up = open[i] > open[i - 1];
            let curr_bullish = close[i] > open[i];
            let curr_marubozu = (close[i] - open[i]) > (high[i] - low[i]) * 0.9;
            prev_bearish && prev_marubozu && gap_up && curr_bullish && curr_marubozu
        }

fn detect_kicker_bear_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bullish = close[i - 1] > open[i - 1];
            let prev_marubozu = (close[i - 1] - open[i - 1]) > (high[i - 1] - low[i - 1]) * 0.9;
            let gap_down = open[i] < open[i - 1];
            let curr_bearish = open[i] > close[i];
            let curr_marubozu = (open[i] - close[i]) > (high[i] - low[i]) * 0.9;
            prev_bullish && prev_marubozu && gap_down && curr_bearish && curr_marubozu
        }

fn detect_hanging_man_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bullish = close[i - 1] > open[i - 1];
            let body = (close[i] - open[i]).abs();
            let lower_shadow = open[i].min(close[i]) - low[i];
            let upper_shadow = high[i] - open[i].max(close[i]);
            let range = high[i] - low[i];
            prev_bullish && range > 0.0 && lower_shadow > body * 2.0 && upper_shadow < body * 0.5
        }

fn detect_doji_star_bullish_at(open: &[f64], high: &[f64], low: &[f64], close: &[f64], i: usize) -> bool {
    
            if i == 0 {
                return false;
            }
            let prev_bearish = open[i - 1] > close[i - 1]
                && (open[i - 1] - close[i - 1]) > (high[i - 1] - low[i - 1]) * 0.5;
            let curr_doji = (close[i] - open[i]).abs() < (high[i] - low[i]) * 0.1;
            let gap_down = open[i].max(close[i]) < close[i - 1];
            prev_bearish && curr_doji && gap_down
        }
