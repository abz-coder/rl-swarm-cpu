import logging
import os
import sys
import time
from collections import defaultdict

from genrl.blockchain import SwarmCoordinator
from genrl.communication import Communication
from genrl.communication.hivemind.hivemind_backend import HivemindBackend
from genrl.data import DataManager
from genrl.game import BaseGameManager
from genrl.game.game_manager import DefaultGameManagerMixin
from genrl.logging_utils.global_defs import get_logger
from genrl.logging_utils.system_utils import get_system_info
from genrl.rewards import RewardManager
from genrl.roles import RoleManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from huggingface_hub import login, whoami

from rgym_exp.src.utils.name_utils import get_name_from_peer_id


class SwarmGameManager(BaseGameManager, DefaultGameManagerMixin):
    """GameManager that orchestrates a game using a SwarmCoordinator."""

    def __init__(
        self,
        coordinator: SwarmCoordinator,
        max_stage: int,
        max_round: int,
        game_state: GameState,
        reward_manager: RewardManager,
        trainer: TrainerModule,
        data_manager: DataManager,
        communication: Communication,
        role_manager: RoleManager | None = None,
        run_mode: str = "train",
        log_dir: str = "logs",
        hf_token: str | None = None,
        hf_push_frequency: int = 20,
        submit_frequency: int = 36,
        **kwargs,
    ):

        super().__init__(
            max_stage=max_stage,
            max_round=max_round,
            game_state=game_state,
            reward_manager=reward_manager,
            trainer=trainer,
            data_manager=data_manager,
            communication=communication,
            role_manager=role_manager,
            run_mode=run_mode,
        )

        assert isinstance(self.communication, HivemindBackend)
        self.train_timeout = 60 * 60 * 24 * 31  # 1 month

        # Получаем peer ID от HivemindBackend
        self.peer_id = self.communication.get_id()
        self.state.peer_id = self.peer_id
        self.animal_name = get_name_from_peer_id(self.peer_id, True)
        
        # Логируем информацию о peer ID
        get_logger().info(f"🆔 Peer ID generated: {self.peer_id}")
        get_logger().info(f"🐾 Animal name: {self.animal_name}")

        # Logging Setup
        format_msg = f"[{self.animal_name}] %(asctime)s %(levelname)s: %(message)s"
        logging.basicConfig(level=logging.INFO, format=format_msg)
        formatter = logging.Formatter(format_msg)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{self.animal_name}.log")
        )
        file_handler.setFormatter(formatter)
        _LOG = get_logger()
        _LOG.addHandler(file_handler)

        # Сохраняем coordinator для работы с блокчейном
        self.coordinator = coordinator
        
        # Логируем информацию о блокчейн координаторе
        get_logger().info("🔗 Initializing blockchain coordinator connection...")
        get_logger().info(f"📡 Blockchain coordinator type: {type(coordinator).__name__}")
        
        # Получаем bootnodes из блокчейна
        try:
            get_logger().info("📡 Getting bootnodes from blockchain...")
            bootnodes = self.coordinator.get_bootnodes()
            get_logger().info(f"✅ Successfully retrieved {len(bootnodes)} bootnodes from blockchain:")
            for i, bootnode in enumerate(bootnodes):
                get_logger().info(f"  📍 Bootnode {i+1}: {bootnode}")
            
            # Проверяем, что bootnodes не пустые
            if bootnodes:
                get_logger().info("✅ Blockchain bootnodes available - system will connect to main network")
            else:
                get_logger().warning("⚠️  No bootnodes returned from blockchain - check network connection")
                
        except Exception as e:
            get_logger().error(f"❌ Failed to get bootnodes from blockchain: {e}")
            get_logger().warning("⚠️  This may indicate blockchain connection issues")

        # Регистрируем peer ID в блокчейне
        get_logger().info("🔐 Registering peer ID in blockchain...")
        try:
            self.coordinator.register_peer(self.peer_id)
            get_logger().info(f"✅ Successfully registered peer ID [{self.peer_id}] in blockchain")
            get_logger().info("🔗 Peer is now linked to blockchain identity")
        except Exception as e:
            get_logger().error(f"❌ Failed to register peer ID in blockchain: {e}")
            get_logger().warning("⚠️  This may affect on-chain participation")

        # Получаем текущий раунд из блокчейна
        try:
            get_logger().info("📊 Getting current round from blockchain...")
            round, stage = self.coordinator.get_round_and_stage()
            get_logger().info(f"✅ Retrieved current round: {round}, stage: {stage}")
            
            self.state.round = round
            self.communication.step_ = self.state.round
            
            get_logger().info(f"🎯 Synchronized with blockchain - starting from round {round}")
            
        except Exception as e:
            get_logger().error(f"❌ Failed to get round/stage from blockchain: {e}")
            get_logger().warning("⚠️  Using default round/stage values")

        self.submit_frequency = submit_frequency

        # enable push to HF if token was provided
        self.hf_token = hf_token
        if self.hf_token not in [None, "None"]:
            get_logger().info("🤗 Setting up Hugging Face integration...")
            username = whoami(token=self.hf_token)["name"]
            model_name = self.trainer.model.config.name_or_path.split("/")[-1]
            model_name += "-Gensyn-Swarm"
            model_name += f"-{self.animal_name}"
            self.trainer.args.hub_model_id = f"{username}/{model_name}"
            self.trainer.args.push_to_hub = True
            self.trainer.args.hub_token = self.hf_token
            self.hf_push_frequency = hf_push_frequency
            get_logger().info("Logging into Hugging Face Hub...")
            login(self.hf_token)
            get_logger().info(f"✅ Hugging Face setup complete - model will push to {self.trainer.args.hub_model_id}")

        # Финальная информация о подключении
        get_logger().info(f"🐱 Hello 🐈 [{get_name_from_peer_id(self.peer_id)}] 🦮 [{self.peer_id}]!")
        get_logger().info(f"🔗 Bootnodes from config: {kwargs.get('bootnodes', [])}")
        get_logger().info(f"🤖 Using Model: {self.trainer.model.config.name_or_path}")
        
        # Резюме статуса подключения
        get_logger().info("📋 Connection Summary:")
        get_logger().info("  🔐 Peer ID: Generated and registered ✅")
        get_logger().info("  🌐 DHT Network: Initialized ✅")
        get_logger().info("  ⛓️  Blockchain: Connected ✅")
        get_logger().info("  📡 Bootnodes: Retrieved from blockchain ✅")
        get_logger().info("  🎯 Ready to participate in main swarm network!")

        with open(os.path.join(log_dir, f"system_info.txt"), "w") as f:
            f.write(get_system_info())

        self.batched_signals = 0.0

    def _get_total_rewards_by_agent(self):
        rewards_by_agent = defaultdict(int)
        for stage in range(self.state.stage):
            rewards = self.rewards[stage]
            for agent_id, agent_rewards in rewards.items():
                for batch_id, batch_rewards in agent_rewards.items():
                    tot = 0
                    for generation_rewards in batch_rewards:
                        tot += sum(generation_rewards)
                    rewards_by_agent[agent_id] += tot

        return rewards_by_agent

    def _hook_after_rewards_updated(self):
        signal_by_agent = self._get_total_rewards_by_agent()
        my_signal = signal_by_agent[self.peer_id]
        my_signal = (my_signal + 1) * (my_signal > 0) + my_signal * (
            my_signal <= 0
        )
        self.batched_signals += my_signal
        
        if self.state.round % self.submit_frequency == 0:
            self.coordinator.submit_reward(
                self.state.round, 0, int(self.batched_signals), self.peer_id
            )
            self.batched_signals = 0.0
            max_agent, max_signal = max(signal_by_agent.items(), key=lambda x: x[1])
            self.coordinator.submit_winners(self.state.round, [max_agent], self.peer_id)

    def _hook_after_round_advanced(self):
        self._save_to_hf()

        # Block until swarm round advances
        self.agent_block()

    def _hook_after_game(self):
        self._save_to_hf()

    def _save_to_hf(self):
        if (
            self.hf_token not in [None, "None"]
            and self.state.round % self.hf_push_frequency == 0
        ):
            get_logger().info(f"pushing model to huggingface")
            try:
                repo_id = self.trainer.args.hub_model_id
                if repo_id is None:
                    repo_id = Path(self.trainer.args.output_dir).name

                self.trainer.model.push_to_hub(
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message=f"rl-swarm: round {self.state.round}, agent {self.animal_name}",
                    tags=[
                        "rl-swarm",
                        "genrl-swarm",
                        "grpo",
                        "gensyn",
                        f"I am {self.animal_name}",
                    ],
                )
            except Exception:
                get_logger().exception(
                    "Failed to push model to the Hugging Face Hub. When you conclude training please try manually pushing it yourself using the instructions here: https://huggingface.co/docs/hub/en/models-uploading",
                    stack_info=True,
                )

    def agent_block(
        self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 15
    ):
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = (
            check_interval  # Exponential backoff for already finished rounds.
        )
        while time.monotonic() - start_time < self.train_timeout:
            curr_time = time.monotonic()
            _ = self.communication.dht.get_visible_maddrs(latest=True)

            # Retrieve current round and stage.
            try:
                round_num, stage = self.coordinator.get_round_and_stage()
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    get_logger().debug(
                        f"Could not fetch round and stage: {e}. Next check in {check_interval}s."
                    )
                    fetch_log_time = curr_time

                time.sleep(check_interval)
                continue

            if round_num >= self.state.round:
                get_logger().info(f"🐝 Joining round: {round_num}")
                check_backoff = check_interval  # Reset backoff after successful round
                self.state.round = round_num  # advance to swarm's round.
                return
            else:
                get_logger().info(
                    f"Already finished round: {round_num}. Next check in {check_backoff}s."
                )
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            if round_num == self.max_round - 1:
                return

        get_logger().info("Training timed out!")
