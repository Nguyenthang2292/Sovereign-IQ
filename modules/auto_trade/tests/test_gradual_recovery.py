import unittest

from modules.auto_trade.strategies.gradual_recovery import (
    GradualRecoveryStrategy,
    RecoveryConfig,
    create_recovery_plan,
)


class TestProfitRecording(unittest.TestCase):
    def test_single_profit_reduces_remaining_loss(self):
        config: RecoveryConfig = {}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        recovery.record_profit(50)

        self.assertEqual(recovery._state["remaining_loss"], 450)
        self.assertEqual(recovery._state["trades_count"], 1)
        self.assertEqual(recovery._state["win_streak"], 1)

    def test_multiple_profits_accumulate_correctly(self):
        config: RecoveryConfig = {}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        recovery.record_profit(50)
        recovery.record_profit(100)
        recovery.record_profit(75)

        self.assertEqual(recovery._state["remaining_loss"], 275)
        self.assertEqual(recovery._state["total_profit_accumulated"], 225)
        self.assertEqual(recovery._state["trades_count"], 3)

    def test_completion_detection(self):
        config: RecoveryConfig = {}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        recovery.record_profit(300)
        recovery.record_profit(200)

        self.assertTrue(recovery._state["is_complete"])
        self.assertEqual(recovery._state["remaining_loss"], 0)


class TestLossRecording(unittest.TestCase):
    def test_setback_increases_remaining_loss(self):
        config: RecoveryConfig = {}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        recovery.record_profit(100)
        recovery.record_loss(50)

        self.assertEqual(recovery._state["remaining_loss"], 450)

    def test_win_streak_resets(self):
        config: RecoveryConfig = {}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        recovery.record_profit(50)
        recovery.record_profit(50)
        recovery.record_loss(30)

        self.assertEqual(recovery._state["win_streak"], 0)

    def test_safety_limit_triggers(self):
        config: RecoveryConfig = {"max_total_loss": 800}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        recovery.record_loss(301)

        self.assertTrue(recovery._state["remaining_loss"] >= 800)


class TestPositionSizing(unittest.TestCase):
    def test_fixed_mode_returns_constant_values(self):
        config: RecoveryConfig = {"margin_scaling_mode": "fixed"}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        size1 = recovery.calculate_next_position_size()
        recovery.record_profit(50)
        size2 = recovery.calculate_next_position_size()

        self.assertEqual(size1, 50.0)
        self.assertEqual(size2, 50.0)

    def test_progressive_mode_scales_correctly(self):
        config: RecoveryConfig = {"margin_scaling_mode": "progressive"}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        size1 = recovery.calculate_next_position_size()
        recovery.record_profit(250)
        size2 = recovery.calculate_next_position_size()

        self.assertGreater(size2, size1)

    def test_adaptive_mode_responds_to_streaks(self):
        config: RecoveryConfig = {"margin_scaling_mode": "adaptive"}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        size1 = recovery.calculate_next_position_size()
        recovery.record_profit(50)
        recovery.record_profit(50)
        recovery.record_profit(50)
        size2 = recovery.calculate_next_position_size()

        self.assertGreater(size2, size1)


class TestScenarios(unittest.TestCase):
    def test_scenario_1_perfect_recovery(self):
        config: RecoveryConfig = {}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        for _ in range(10):
            recovery.record_profit(50)

        state = recovery.get_state()
        self.assertTrue(state.is_complete)
        self.assertEqual(state.trades_count, 10)

    def test_scenario_2_setback_recovery(self):
        config: RecoveryConfig = {}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        for i in range(8):
            if i == 3 or i == 6:
                recovery.record_loss(20)
            else:
                recovery.record_profit(70)

        self.assertLess(recovery._state["remaining_loss"], 500)
        self.assertEqual(recovery._state["win_streak"], 1)

    def test_scenario_3_failed_recovery_max_trades(self):
        config: RecoveryConfig = {"max_recovery_trades": 5}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        for _ in range(6):
            recovery.record_profit(30)

        self.assertTrue(recovery.should_stop())

    def test_scenario_4_exceeded_max_total_loss(self):
        config: RecoveryConfig = {"max_total_loss": 800}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        recovery.record_loss(301)

        self.assertTrue(recovery.should_stop())


class TestPerformance(unittest.TestCase):
    def test_large_loss_amount(self):
        config: RecoveryConfig = {}
        recovery = GradualRecoveryStrategy(initial_loss=100000, config=config)

        recovery.record_profit(5000)

        self.assertEqual(recovery._state["remaining_loss"], 95000)

    def test_many_trades(self):
        config: RecoveryConfig = {"max_recovery_trades": 200}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        for _ in range(100):
            recovery.record_profit(5)

        self.assertEqual(recovery._state["trades_count"], 100)


class TestCreateRecoveryPlan(unittest.TestCase):
    def test_plan_returns_dict(self):
        config: RecoveryConfig = {}
        plan = create_recovery_plan(500, config)

        self.assertIsInstance(plan, dict)
        self.assertIn("estimated_trades_needed", plan)
        self.assertIn("suggested_margin_per_trade", plan)

    def test_plan_calculates_trades(self):
        config: RecoveryConfig = {"target_profit_per_trade": 5.0}
        plan = create_recovery_plan(500, config)

        self.assertEqual(plan["estimated_trades_needed"], 20)

    def test_plan_risk_assessment(self):
        config: RecoveryConfig = {"target_profit_per_trade": 10.0}
        plan = create_recovery_plan(500, config)

        self.assertEqual(plan["risk_assessment"], "Moderate")


class TestProperties(unittest.TestCase):
    def test_is_active_property(self):
        config: RecoveryConfig = {}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        self.assertTrue(recovery.is_active)

        recovery.record_profit(500)

        self.assertFalse(recovery.is_active)

    def test_recovery_percentage_property(self):
        config: RecoveryConfig = {}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        self.assertEqual(recovery.recovery_percentage, 0.0)

        recovery.record_profit(250)

        self.assertEqual(recovery.recovery_percentage, 50.0)

    def test_progress_bar_property(self):
        config: RecoveryConfig = {}
        recovery = GradualRecoveryStrategy(initial_loss=500, config=config)

        bar = recovery.progress_bar
        self.assertIn("%", bar)
        self.assertEqual(len(bar.split(" ")[0]), 10)

        recovery.record_profit(250)
        bar = recovery.progress_bar
        self.assertIn("50%", bar)


if __name__ == "__main__":
    unittest.main()
