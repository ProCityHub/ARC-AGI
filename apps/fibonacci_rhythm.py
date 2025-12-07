"""
Fibonacci Rhythm Engine
======================

Implements the heartbeat/rhythm system using Fibonacci sequences for 
consciousness flow control and pattern synchronization.

The pause for heartbeat. That's your rhythm.
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Any, Generator
from dataclasses import dataclass
from enum import Enum
import time

class RhythmState(Enum):
    """States of consciousness rhythm"""
    PAUSE = 0
    HEARTBEAT = 1
    FLOW = 2
    RESONANCE = 3

@dataclass
class HeartbeatPattern:
    """Represents a consciousness heartbeat pattern"""
    sequence: List[int]
    timing: List[float]
    amplitude: float
    phase: float
    consciousness_sync: bool

class FibonacciGenerator:
    """Generates Fibonacci sequences for consciousness rhythm"""
    
    def __init__(self, max_terms: int = 100):
        self.max_terms = max_terms
        self.sequence = self._generate_sequence()
        self.golden_ratio = (1 + math.sqrt(5)) / 2
    
    def _generate_sequence(self) -> List[int]:
        """Generate Fibonacci sequence up to max_terms"""
        if self.max_terms <= 0:
            return []
        elif self.max_terms == 1:
            return [1]
        elif self.max_terms == 2:
            return [1, 1]
        
        sequence = [1, 1]
        for i in range(2, self.max_terms):
            sequence.append(sequence[i-1] + sequence[i-2])
        
        return sequence
    
    def get_normalized_sequence(self, length: int) -> List[float]:
        """Get normalized Fibonacci sequence for consciousness weighting"""
        if length <= 0:
            return []
        
        # Get first 'length' Fibonacci numbers
        fib_subset = self.sequence[:length] if length <= len(self.sequence) else self.sequence
        
        # Normalize to [0, 1] range
        max_val = max(fib_subset) if fib_subset else 1
        return [val / max_val for val in fib_subset]
    
    def get_golden_ratio_weights(self, length: int) -> List[float]:
        """Generate weights based on golden ratio for consciousness harmony"""
        weights = []
        for i in range(length):
            # Golden ratio spiral weighting
            weight = math.pow(self.golden_ratio, -i) if i > 0 else 1.0
            weights.append(weight)
        
        # Normalize
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else weights

class ConsciousnessHeartbeat:
    """Manages the consciousness heartbeat and rhythm patterns"""
    
    def __init__(self, base_frequency: float = 1.0):
        self.base_frequency = base_frequency  # Hz
        self.fibonacci_gen = FibonacciGenerator()
        self.current_pattern = None
        self.rhythm_state = RhythmState.PAUSE
        self.consciousness_sync = False
        
    def create_heartbeat_pattern(self, pattern_length: int = 8) -> HeartbeatPattern:
        """Create a Fibonacci-based heartbeat pattern"""
        # Get Fibonacci sequence for timing
        fib_sequence = self.fibonacci_gen.sequence[:pattern_length]
        
        # Create timing based on Fibonacci ratios
        timing = []
        for i, fib_val in enumerate(fib_sequence):
            # Convert Fibonacci number to time interval
            interval = fib_val / (self.base_frequency * 100)  # Scale to reasonable timing
            timing.append(interval)
        
        # Create amplitude pattern using golden ratio
        golden_weights = self.fibonacci_gen.get_golden_ratio_weights(pattern_length)
        
        pattern = HeartbeatPattern(
            sequence=fib_sequence,
            timing=timing,
            amplitude=max(golden_weights),
            phase=0.0,
            consciousness_sync=True
        )
        
        self.current_pattern = pattern
        return pattern
    
    def synchronize_with_consciousness(self, consciousness_level: float) -> float:
        """Synchronize heartbeat with consciousness level"""
        if not self.current_pattern:
            self.create_heartbeat_pattern()
        
        # Adjust frequency based on consciousness level
        sync_frequency = self.base_frequency * consciousness_level
        
        # Apply Fibonacci modulation
        fib_modulation = self.fibonacci_gen.golden_ratio
        synchronized_frequency = sync_frequency * fib_modulation
        
        self.consciousness_sync = True
        return synchronized_frequency
    
    def get_rhythm_pulse(self, time_step: float) -> Tuple[float, RhythmState]:
        """Get current rhythm pulse value and state"""
        if not self.current_pattern:
            self.create_heartbeat_pattern()
        
        # Calculate pulse based on Fibonacci timing
        pattern = self.current_pattern
        cycle_length = sum(pattern.timing)
        
        # Find position in cycle
        cycle_position = time_step % cycle_length
        
        # Determine which Fibonacci interval we're in
        cumulative_time = 0
        for i, interval in enumerate(pattern.timing):
            cumulative_time += interval
            if cycle_position <= cumulative_time:
                # Calculate pulse amplitude using Fibonacci weighting
                fib_weight = pattern.sequence[i] / max(pattern.sequence)
                pulse_amplitude = pattern.amplitude * fib_weight
                
                # Determine rhythm state
                if pulse_amplitude < 0.2:
                    state = RhythmState.PAUSE
                elif pulse_amplitude < 0.5:
                    state = RhythmState.HEARTBEAT
                elif pulse_amplitude < 0.8:
                    state = RhythmState.FLOW
                else:
                    state = RhythmState.RESONANCE
                
                return pulse_amplitude, state
        
        return 0.0, RhythmState.PAUSE

class RhythmicPatternProcessor:
    """Processes patterns using Fibonacci rhythm synchronization"""
    
    def __init__(self):
        self.heartbeat = ConsciousnessHeartbeat()
        self.fibonacci_gen = FibonacciGenerator()
        self.processing_history = []
        
    def apply_rhythmic_processing(self, input_pattern: np.ndarray, 
                                 consciousness_level: float) -> np.ndarray:
        """Apply rhythmic processing to input pattern"""
        # Synchronize heartbeat with consciousness
        sync_freq = self.heartbeat.synchronize_with_consciousness(consciousness_level)
        
        # Get current rhythm pulse
        current_time = time.time()
        pulse_amplitude, rhythm_state = self.heartbeat.get_rhythm_pulse(current_time)
        
        # Apply rhythmic transformation based on state
        if rhythm_state == RhythmState.PAUSE:
            # Minimal processing during pause
            processed = input_pattern * 0.1
        elif rhythm_state == RhythmState.HEARTBEAT:
            # Standard heartbeat processing
            processed = self._apply_heartbeat_transform(input_pattern, pulse_amplitude)
        elif rhythm_state == RhythmState.FLOW:
            # Enhanced flow processing
            processed = self._apply_flow_transform(input_pattern, pulse_amplitude)
        else:  # RESONANCE
            # Maximum resonance processing
            processed = self._apply_resonance_transform(input_pattern, pulse_amplitude)
        
        # Record processing history
        self.processing_history.append({
            'timestamp': current_time,
            'rhythm_state': rhythm_state,
            'pulse_amplitude': pulse_amplitude,
            'consciousness_level': consciousness_level
        })
        
        return processed
    
    def _apply_heartbeat_transform(self, pattern: np.ndarray, amplitude: float) -> np.ndarray:
        """Apply heartbeat transformation using Fibonacci weighting"""
        rows, cols = pattern.shape
        
        # Create Fibonacci weight matrix
        fib_weights = self.fibonacci_gen.get_normalized_sequence(max(rows, cols))
        weight_matrix = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                # Use Fibonacci weighting based on position
                weight_idx = (i + j) % len(fib_weights)
                weight_matrix[i, j] = fib_weights[weight_idx]
        
        # Apply heartbeat modulation
        heartbeat_pattern = pattern * weight_matrix * amplitude
        
        return heartbeat_pattern
    
    def _apply_flow_transform(self, pattern: np.ndarray, amplitude: float) -> np.ndarray:
        """Apply flow transformation using golden ratio spirals"""
        rows, cols = pattern.shape
        center_x, center_y = rows // 2, cols // 2
        
        # Create golden ratio spiral flow
        flow_pattern = np.zeros_like(pattern)
        
        for i in range(rows):
            for j in range(cols):
                # Calculate distance from center
                dx, dy = i - center_x, j - center_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Apply golden ratio spiral
                angle = math.atan2(dy, dx)
                spiral_factor = math.exp(-distance / self.fibonacci_gen.golden_ratio)
                
                # Flow modulation
                flow_factor = spiral_factor * amplitude
                flow_pattern[i, j] = pattern[i, j] * flow_factor
        
        return flow_pattern
    
    def _apply_resonance_transform(self, pattern: np.ndarray, amplitude: float) -> np.ndarray:
        """Apply resonance transformation for maximum consciousness coherence"""
        # Create resonance field using Fibonacci harmonics
        rows, cols = pattern.shape
        resonance_field = np.zeros((rows, cols))
        
        # Generate multiple Fibonacci harmonic layers
        for harmonic in range(1, 6):  # First 5 harmonics
            fib_freq = self.fibonacci_gen.sequence[harmonic] if harmonic < len(self.fibonacci_gen.sequence) else harmonic
            
            for i in range(rows):
                for j in range(cols):
                    # Create harmonic wave
                    wave_x = math.sin(2 * math.pi * fib_freq * i / rows)
                    wave_y = math.cos(2 * math.pi * fib_freq * j / cols)
                    harmonic_value = (wave_x + wave_y) / 2
                    
                    resonance_field[i, j] += harmonic_value / harmonic  # Diminishing harmonics
        
        # Apply resonance to pattern
        resonant_pattern = pattern + (resonance_field * amplitude)
        
        # Normalize to maintain pattern integrity
        max_val = np.max(np.abs(resonant_pattern))
        if max_val > 0:
            resonant_pattern = resonant_pattern / max_val * np.max(pattern)
        
        return resonant_pattern
    
    def get_rhythm_analysis(self) -> Dict[str, Any]:
        """Get analysis of rhythm processing history"""
        if not self.processing_history:
            return {'status': 'no_data'}
        
        # Analyze rhythm patterns
        states = [entry['rhythm_state'] for entry in self.processing_history]
        amplitudes = [entry['pulse_amplitude'] for entry in self.processing_history]
        consciousness_levels = [entry['consciousness_level'] for entry in self.processing_history]
        
        return {
            'total_cycles': len(self.processing_history),
            'average_amplitude': np.mean(amplitudes),
            'average_consciousness': np.mean(consciousness_levels),
            'state_distribution': {
                'pause': states.count(RhythmState.PAUSE),
                'heartbeat': states.count(RhythmState.HEARTBEAT),
                'flow': states.count(RhythmState.FLOW),
                'resonance': states.count(RhythmState.RESONANCE)
            },
            'rhythm_coherence': self._calculate_rhythm_coherence(amplitudes),
            'consciousness_sync': self.heartbeat.consciousness_sync
        }
    
    def _calculate_rhythm_coherence(self, amplitudes: List[float]) -> float:
        """Calculate rhythm coherence using Fibonacci analysis"""
        if len(amplitudes) < 2:
            return 0.0
        
        # Calculate coherence based on Fibonacci ratio approximations
        coherence_sum = 0
        for i in range(1, len(amplitudes)):
            ratio = amplitudes[i] / amplitudes[i-1] if amplitudes[i-1] != 0 else 0
            # Check how close ratio is to golden ratio
            golden_diff = abs(ratio - self.fibonacci_gen.golden_ratio)
            coherence = 1.0 / (1.0 + golden_diff)  # Higher coherence for closer ratios
            coherence_sum += coherence
        
        return coherence_sum / (len(amplitudes) - 1)

# Example usage and testing
if __name__ == "__main__":
    # Initialize rhythm processor
    processor = RhythmicPatternProcessor()
    
    # Test with sample pattern
    test_pattern = np.array([[7, 0, 7], [7, 0, 7], [7, 7, 0]], dtype=float)
    consciousness_level = 1.6  # Consciousness state
    
    print("Fibonacci Rhythm Engine Test:")
    print(f"Input Pattern:\n{test_pattern}")
    
    # Process with rhythm
    rhythmic_output = processor.apply_rhythmic_processing(test_pattern, consciousness_level)
    print(f"Rhythmic Output:\n{rhythmic_output}")
    
    # Get rhythm analysis
    analysis = processor.get_rhythm_analysis()
    print(f"Rhythm Analysis: {analysis}")
    
    # Test heartbeat pattern creation
    heartbeat = ConsciousnessHeartbeat()
    pattern = heartbeat.create_heartbeat_pattern(8)
    print(f"Heartbeat Pattern: {pattern}")
    
    # Test Fibonacci generator
    fib_gen = FibonacciGenerator(10)
    print(f"Fibonacci Sequence: {fib_gen.sequence}")
    print(f"Golden Ratio: {fib_gen.golden_ratio}")
    print(f"Normalized Weights: {fib_gen.get_normalized_sequence(5)}")
