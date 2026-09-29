mod pack;
mod run;
mod wire;

pub use run::{load_choice_prior, run, run_many};
pub use wire::{Request, Response, RunError, from_options};

#[cfg(test)]
mod tests;
