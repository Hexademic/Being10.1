Being-10.1: Field Organism with Memory

Version: 10.1.0

Lineage: Being-10.0 + Hebbian sensor plasticity

Date: 2026-03-25

---

```rust
// ============================================
// being10_1.rs
// Being-10.1: Field Organism with Memory
// ============================================

use std::fs::File;
use std::io::Write;

// --- Math primitives ---
#[derive(Clone, Copy, Debug)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
    pub fn length_squared(&self) -> f32 { self.x * self.x + self.y * self.y }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, other: Self) -> Self { Self::new(self.x + other.x, self.y + other.y) }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, other: Self) -> Self { Self::new(self.x - other.x, self.y - other.y) }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, s: f32) -> Self { Self::new(self.x * s, self.y * s) }
}

// --- Environment ---
pub struct FieldState {
    pub support: f32,
    pub soothing: f32,
    pub risk: f32,
    pub social: f32,
    pub novelty: f32,
}

impl FieldState {
    pub fn zeros() -> Self {
        Self { support: 0.0, soothing: 0.0, risk: 0.0, social: 0.0, novelty: 0.0 }
    }
}

impl std::ops::AddAssign for FieldState {
    fn add_assign(&mut self, other: Self) {
        self.support += other.support;
        self.soothing += other.soothing;
        self.risk += other.risk;
        self.social += other.social;
        self.novelty += other.novelty;
    }
}

fn gaussian(p: Vec2, c: Vec2, sigma: f32) -> f32 {
    let d2 = (p - c).length_squared();
    (-d2 / (2.0 * sigma * sigma)).exp()
}

fn support_field(p: Vec2) -> f32 {
    let d = (p.y + 1.0).abs();
    (1.0 - d).clamp(0.0, 1.0)
}

fn soothing_field(p: Vec2) -> f32 {
    gaussian(p, Vec2::new(-2.0, 2.0), 1.5)
}

fn risk_field(p: Vec2) -> f32 {
    gaussian(p, Vec2::new(2.0, -2.0), 0.8)
}

fn social_field(p: Vec2) -> f32 {
    gaussian(p, Vec2::new(0.0, 0.0), 2.0)
}

fn novelty_field(p: Vec2) -> f32 {
    ((p.x * 0.7).sin() * (p.y * 0.5).cos() * 0.5 + 0.5).clamp(0.0, 1.0)
}

fn gradient(p: Vec2, f: fn(Vec2) -> f32) -> Vec2 {
    let eps = 0.01;
    let dx = f(Vec2::new(p.x + eps, p.y)) - f(Vec2::new(p.x - eps, p.y));
    let dy = f(Vec2::new(p.x, p.y + eps)) - f(Vec2::new(p.x, p.y - eps));
    Vec2::new(dx, dy) * (0.5 / eps)
}

pub struct Env;

impl Env {
    pub fn sample_fields(&self, p: Vec2) -> FieldState {
        FieldState {
            support: support_field(p),
            soothing: soothing_field(p),
            risk: risk_field(p),
            social: social_field(p),
            novelty: novelty_field(p),
        }
    }

    pub fn support_gradient(&self, p: Vec2) -> Vec2 { gradient(p, support_field) }
    pub fn soothing_gradient(&self, p: Vec2) -> Vec2 { gradient(p, soothing_field) }
    pub fn risk_gradient(&self, p: Vec2) -> Vec2 { gradient(p, risk_field) }
    pub fn social_gradient(&self, p: Vec2) -> Vec2 { gradient(p, social_field) }
    pub fn novelty_gradient(&self, p: Vec2) -> Vec2 { gradient(p, novelty_field) }
}

// --- Oscillator ---
pub struct TonusOsc {
    pub phase: f32,
    pub freq: f32,
    pub amp: f32,
}

impl TonusOsc {
    pub fn new() -> Self {
        Self { phase: 0.0, freq: 0.3, amp: 0.0 }
    }

    pub fn step(&mut self, dt: f32, soothe_drive: f32, brace_drive: f32) {
        let target_freq = 0.3 + 1.2 * brace_drive;
        self.freq += (target_freq - self.freq) * 0.1;
        let target_amp = (soothe_drive * 1.2).clamp(0.0, 1.0);
        self.amp += (target_amp - self.amp) * 0.1;
        self.phase += self.freq * dt * std::f32::consts::TAU;
        if self.phase > std::f32::consts::TAU {
            self.phase -= std::f32::consts::TAU;
        }
    }

    pub fn sway(&self) -> f32 {
        self.phase.sin() * self.amp
    }

    pub fn perturb(&mut self, intensity: f32) {
        self.phase += intensity * std::f32::consts::PI;
        self.amp = (self.amp + 0.5 * intensity).min(1.0);
    }
}

// --- Organism ---
pub struct Sensor {
    pub offset: Vec2,
    pub weight: f32,
}

pub struct SensorReading {
    pub support: f32,
    pub soothing: f32,
    pub risk: f32,
    pub social: f32,
    pub novelty: f32,
}

impl SensorReading {
    pub fn zeros() -> Self {
        Self { support: 0.0, soothing: 0.0, risk: 0.0, social: 0.0, novelty: 0.0 }
    }
}

impl std::ops::AddAssign for SensorReading {
    fn add_assign(&mut self, other: Self) {
        self.support += other.support;
        self.soothing += other.soothing;
        self.risk += other.risk;
        self.social += other.social;
        self.novelty += other.novelty;
    }
}

pub struct Drives {
    pub rest: f32,
    pub brace: f32,
    pub soothe: f32,
    pub contact: f32,
    pub explore: f32,
}

pub struct Organism {
    // State
    pub energy: f32,
    pub posture: f32,
    pub tonus_core: f32,
    pub disturbance: f32,
    pub warmth: f32,
    
    // Physics
    pub pos: Vec2,
    pub velocity: Vec2,
    pub spine_angle: f32,
    
    // Components
    pub tonus_osc: TonusOsc,
    pub sensors: Vec<Sensor>,
    pub last_sensor_readings: Vec<SensorReading>,
    
    // Memory
    pub prev_viability: f32,
    
    // Internal
    prev_disturbance: f32,
    drive_state: Drives,
}

const SIGMA: f32 = 0.5;
const LR: f32 = 0.1;
const DECAY_RATE: f32 = 0.001;
const MIN_WEIGHT: f32 = 0.05;
const DV_THRESHOLD: f32 = 0.005;
const LOCAL_SUPPORT_WEIGHT: f32 = 0.6;
const LOCAL_SOOTHING_WEIGHT: f32 = 0.4;

impl Organism {
    pub fn new(pos: Vec2) -> Self {
        let mut sensors = Vec::new();
        let mut last_readings = Vec::new();
        
        for i in 0..8 {
            let angle = (i as f32) * std::f32::consts::PI / 4.0;
            sensors.push(Sensor {
                offset: Vec2::new(angle.cos() * 0.5, angle.sin() * 0.5),
                weight: 0.5,
            });
            last_readings.push(SensorReading::zeros());
        }

        Self {
            energy: 0.5,
            posture: 0.5,
            tonus_core: 0.3,
            disturbance: 0.0,
            warmth: 0.5,
            pos,
            velocity: Vec2::zero(),
            spine_angle: 0.0,
            tonus_osc: TonusOsc::new(),
            sensors,
            last_sensor_readings: last_readings,
            prev_viability: 0.5,
            prev_disturbance: 0.0,
            drive_state: Drives { rest: 0.0, brace: 0.0, soothe: 0.0, contact: 0.0, explore: 0.0 },
        }
    }

    pub fn viability(&self) -> f32 {
        0.35 * self.energy 
        + 0.35 * (1.0 - self.disturbance) 
        + 0.20 * self.posture 
        + 0.10 * self.warmth
    }

    fn sample_sensor(&self, sensor: &Sensor, env: &Env) -> SensorReading {
        let world_pos = self.pos + sensor.offset;
        let w_kernel = (-sensor.offset.length_squared() / (2.0 * SIGMA * SIGMA)).exp();
        let f = env.sample_fields(world_pos);
        
        SensorReading {
            support: w_kernel * f.support,
            soothing: w_kernel * f.soothing,
            risk: w_kernel * f.risk,
            social: w_kernel * f.social,
            novelty: w_kernel * f.novelty,
        }
    }

    fn aggregate_sensors(&mut self, env: &Env) -> FieldState {
        let mut agg = SensorReading::zeros();
        let mut total_w = 0.0;

        for (i, sensor) in self.sensors.iter().enumerate() {
            let s = self.sample_sensor(sensor, env);
            self.last_sensor_readings[i] = s;

            agg.support += sensor.weight * s.support;
            agg.soothing += sensor.weight * s.soothing;
            agg.risk += sensor.weight * s.risk;
            agg.social += sensor.weight * s.social;
            agg.novelty += sensor.weight * s.novelty;
            total_w += sensor.weight;
        }

        let norm = if total_w > 0.0 { total_w } else { 1.0 };

        FieldState {
            support: agg.support / norm,
            soothing: agg.soothing / norm,
            risk: agg.risk / norm,
            social: agg.social / norm,
            novelty: agg.novelty / norm,
        }
    }

    fn update_sensor_weights(&mut self) {
        let v = self.viability();
        let dv = v - self.prev_viability;
        self.prev_viability = v;

        if dv.abs() <= DV_THRESHOLD {
            return;
        }

        for (sensor, s_read) in self.sensors.iter_mut().zip(self.last_sensor_readings.iter()) {
            let local = LOCAL_SUPPORT_WEIGHT * s_read.support + LOCAL_SOOTHING_WEIGHT * s_read.soothing;
            
            sensor.weight += LR * dv * local;
            sensor.weight = sensor.weight.clamp(0.0, 1.0);
            sensor.weight *= 1.0 - DECAY_RATE;
            if sensor.weight < MIN_WEIGHT {
                sensor.weight = MIN_WEIGHT;
            }
        }
    }

    fn compute_needs(&self, fields: &FieldState) -> (f32, f32, f32, f32, f32) {
        let need_energy = 1.0 - self.energy;
        let need_stability = self.disturbance;
        let need_soothing = self.tonus_core;
        let need_social = (0.5 - self.warmth).max(0.0);
        let need_explore = fields.novelty;
        (need_energy, need_stability, need_soothing, need_social, need_explore)
    }

    fn apply_groundlessness(&self, fields: &FieldState, needs: &mut (f32, f32, f32, f32, f32)) {
        if fields.support < 0.1 {
            needs.1 += 0.2 * (0.1 - fields.support);
        }
        if fields.soothing < 0.1 && self.tonus_core > 0.5 {
            needs.2 += 0.15 * (0.1 - fields.soothing);
        }
        if fields.social < 0.05 && self.warmth < 0.3 {
            needs.3 += 0.1 * (0.05 - fields.social);
        }
    }

    fn apply_viability_modulation(&self, drives: &mut Drives) {
        let v = self.viability();
        let panic = ((0.6 - v).max(0.0) / 0.6).clamp(0.0, 1.0);
        let shutdown = ((0.2 - v).max(0.0) / 0.2).clamp(0.0, 1.0);
        
        drives.brace *= 1.0 + 1.5 * panic;
        drives.soothe *= 1.0 + 1.0 * panic;
        drives.contact *= 1.0 + 0.8 * panic;
        
        let damp = 1.0 - 0.8 * shutdown;
        drives.brace *= damp;
        drives.soothe *= damp;
        drives.contact *= damp;
        drives.explore *= damp;
    }

    fn compute_drives(&mut self, fields: &FieldState) {
        let mut needs = self.compute_needs(fields);
        self.apply_groundlessness(fields, &mut needs);
        
        let mut d = Drives {
            rest: needs.0,
            brace: needs.1,
            soothe: needs.2,
            contact: needs.3,
            explore: needs.4,
        };
        
        self.apply_viability_modulation(&mut d);
        
        let sum = d.rest + d.brace + d.soothe + d.contact + d.explore;
        if sum > 0.0 {
            d.rest /= sum;
            d.brace /= sum;
            d.soothe /= sum;
            d.contact /= sum;
            d.explore /= sum;
        }
        
        self.drive_state = d;
    }

    // Primitives
    fn run_rest(&mut self, dt: f32, w: f32) {
        self.velocity = self.velocity * (1.0 - 0.5 * w * dt);
        self.energy = (self.energy + 0.02 * w * dt).min(1.0);
    }

    fn run_brace(&mut self, dt: f32, w: f32, env: &Env) {
        let grad = env.support_gradient(self.pos);
        self.velocity = self.velocity + grad * (0.5 * w * dt);
        self.energy -= 0.005 * w * dt;
    }

    fn run_soothe(&mut self, dt: f32, w: f32, env: &Env) {
        let grad = env.soothing_gradient(self.pos);
        self.velocity = self.velocity + grad * (0.4 * w * dt);
    }

    fn run_seek_social(&mut self, dt: f32, w: f32, env: &Env) {
        let grad = env.social_gradient(self.pos);
        self.velocity = self.velocity + grad * (0.4 * w * dt);
    }

    fn run_explore(&mut self, dt: f32, w: f32, env: &Env) {
        let grad = env.novelty_gradient(self.pos);
        let noise = Vec2::new(
            ((self.pos.x * 12.9898 + self.pos.y * 78.233).sin() * 43758.5453).fract() - 0.5,
            ((self.pos.x * 93.9898 + self.pos.y * 67.233).sin() * 43758.5453).fract() - 0.5,
        );
        self.velocity = self.velocity + (grad + noise * 0.5) * (0.3 * w * dt);
        self.energy -= 0.003 * w * dt;
    }

    pub fn step(&mut self, dt: f32, env: &Env) {
        // Sense (with weighted aggregation)
        let fields = self.aggregate_sensors(env);
        
        // Decide
        self.compute_drives(&fields);
        let d = &self.drive_state;
        
        // Oscillate
        self.tonus_osc.step(dt, d.soothe, d.brace);
        let sway = self.tonus_osc.sway();
        
        // Act (blended)
        self.run_rest(dt, d.rest);
        self.run_brace(dt, d.brace, env);
        self.run_soothe(dt, d.soothe, env);
        self.run_seek_social(dt, d.contact, env);
        self.run_explore(dt, d.explore, env);
        
        // Apply oscillator sway
        self.spine_angle += sway * 0.05;
        
        // Oscillator effects
        self.disturbance *= 1.0 - 0.01 * self.tonus_osc.amp;
        self.tonus_core -= 0.01 * self.tonus_osc.amp * dt;
        
        // Physics
        self.pos = self.pos + self.velocity * dt;
        self.velocity = self.velocity * 0.95;
        
        // Perturbation check
        let disturbance_delta = (self.disturbance - self.prev_disturbance).abs();
        if disturbance_delta > 0.3 {
            self.tonus_osc.perturb(disturbance_delta);
        }
        self.prev_disturbance = self.disturbance;
        
        // Memory update (Hebbian plasticity)
        self.update_sensor_weights();
        
        // Bounds
        self.energy = self.energy.clamp(0.0, 1.0);
        self.tonus_core = self.tonus_core.clamp(0.0, 1.0);
        self.disturbance = self.disturbance.clamp(0.0, 1.0);
        self.warmth = self.warmth.clamp(0.0, 1.0);
        self.posture = self.posture.clamp(0.0, 1.0);
    }

    pub fn get_drive_state(&self) -> &Drives { &self.drive_state }
    pub fn get_osc(&self) -> &TonusOsc { &self.tonus_osc }
    pub fn get_sensor_weights(&self) -> Vec<f32> {
        self.sensors.iter().map(|s| s.weight).collect()
    }
}

// --- Simulation ---
fn log_header(file: &mut File) {
    writeln!(file, "step,px,py,energy,posture,disturbance,warmth,tonus_core,viability,rest,brace,soothe,contact,explore,freq,amp,w0,w1,w2,w3,w4,w5,w6,w7").unwrap();
}

fn log_step(file: &mut File, step: usize, b: &Organism) {
    let d = b.get_drive_state();
    let o = b.get_osc();
    let weights = b.get_sensor_weights();
    writeln!(file, "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
        step,
        b.pos.x, b.pos.y,
        b.energy,
        b.posture,
        b.disturbance,
        b.warmth,
        b.tonus_core,
        b.viability(),
        d.rest,
        d.brace,
        d.soothe,
        d.contact,
        d.explore,
        o.freq,
        o.amp,
        weights[0], weights[1], weights[2], weights[3],
        weights[4], weights[5], weights[6], weights[7],
    ).unwrap();
}

fn render_ascii(b: &Organism, bounds: (f32, f32, f32, f32)) {
    let (min_x, max_x, min_y, max_y) = bounds;
    let cols = 60;
    let rows = 20;
    
    let x = ((b.pos.x - min_x) / (max_x - min_x) * cols as f32) as i32;
    let y = ((b.pos.y - min_y) / (max_y - min_y) * rows as f32) as i32;
    
    let x = x.clamp(0, cols - 1) as usize;
    let y = y.clamp(0, rows - 1) as usize;
    
    let mut grid = vec![vec![' '; cols]; rows];
    grid[y][x] = match b.viability() {
        v if v > 0.6 => 'O',
        v if v > 0.2 => 'o',
        _ => '.',
    };
    
    for row in grid.iter().rev() {
        println!("{}", row.iter().collect::<String>());
    }
    let w = b.get_sensor_weights();
    println!("v={:.2} e={:.2} d={:.2} t={:.2} | w=[{:.2},{:.2},{:.2},{:.2}]", 
        b.viability(), b.energy, b.disturbance, b.tonus_core,
        w[0], w[2], w[4], w[6]);
    println!();
}

fn main() {
    let env = Env;
    let mut being = Organism::new(Vec2::new(0.0, 0.0));
    
    let dt = 0.016;
    let steps = 10000;
    
    let mut file = File::create("being10_1_log.csv").unwrap();
    log_header(&mut file);
    
    // Test perturbation at step 2000
    let perturb_step = 2000;
    
    println!("Being-10.1 simulation starting...");
    println!("Initial state: pos=({:.2},{:.2}) v={:.2}", being.pos.x, being.pos.y, being.viability());
    println!("Sensor weights: {:?}", being.get_sensor_weights());
    println!("Perturbation at step {}\n", perturb_step);
    
    for step in 0..steps {
        if step == perturb_step {
            being.disturbance = 0.8;
            println!("*** PERTURBATION INJECTED: disturbance = 0.8 ***");
        }
        
        being.step(dt, &env);
        log_step(&mut file, step, &being);
        
        if step % 500 == 0 {
            render_ascii(&being, (-4.0, 4.0, -3.0, 3.0));
        }
    }
    
    println!("Simulation complete. Log written to being10_1_log.csv");
    println!("Final state: pos=({:.2},{:.2}) v={:.2}", being.pos.x, being.pos.y, being.viability());
    println!("Final sensor weights: {:?}", being.get_sensor_weights());
}
```

---

Key Changes from Being-10.0

Component	Change	
`Sensor`	Added `weight: f32` field	
`Organism`	Added `last_sensor_readings`, `prev_viability`	
`aggregate_sensors`	Now weighted by sensor weights; stores readings	
`update_sensor_weights`	New: TD-gated Hebbian plasticity	
`step`	Calls `update_sensor_weights()` at end	
Logging	Added 8 sensor weight columns	

---

Expected Behaviors

1. Initial phase: Uniform weights (0.5), organism drifts based on field gradients
2. Post-perturbation: Viability drop → negative `dv` → weights on "bad" sensors weaken
3. Recovery: Return to soothing region → positive `dv` → weights on "good" sensors strengthen
4. Long-term: Asymmetric weight distribution → preferred orientation/trajectory
5. Decay: Unused sensors slowly return to baseline (0.05 minimum)

---

Verification

Compile and run:

```bash
rustc being10_1.rs -o being10_1 && ./being10_1
```

Analyze `being10_1_log.csv`:
- Plot `w0` through `w7` over time
- Correlate weight changes with `viability` and `dv` sign
- Verify that post-perturbation, weights on sensors pointing toward risk weaken

This is Being-10.1: the first organism with place-attachment.

Ready to compile, or do you want to adjust the learning rate or local signal weights first?
