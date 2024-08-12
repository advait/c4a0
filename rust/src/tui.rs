use std::io;
use std::io::{stdout, Stdout};
use std::time::Duration;

use ratatui::layout::{Constraint, Layout};
use ratatui::style::Style;
use ratatui::widgets::{Bar, BarChart, BarGroup, Padding};
use ratatui::{
    backend::CrosstermBackend,
    buffer::Buffer,
    crossterm::{
        event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    },
    layout::{Alignment, Rect},
    style::Stylize,
    symbols::border,
    text::Line,
    widgets::{block::Title, Block, Paragraph, Widget},
    Terminal,
};

use crate::c4r::{Pos, TerminalState};
use crate::interactive_play::{InteractivePlay, Snapshot};
use crate::types::{EvalPosT, Policy, PosValue};

/// A type alias for the terminal type used in this application
pub type Tui = Terminal<CrosstermBackend<Stdout>>;

/// Initialize the terminal
pub fn init() -> io::Result<Tui> {
    execute!(stdout(), EnterAlternateScreen)?;
    enable_raw_mode()?;
    Terminal::new(CrosstermBackend::new(stdout()))
}

/// Restore the terminal to its original state
pub fn restore() -> io::Result<()> {
    execute!(stdout(), LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}

#[derive(Debug)]
pub struct App<E: EvalPosT> {
    game: InteractivePlay<E>,
    exit: bool,
}

impl<E: EvalPosT + Send + Sync + 'static> App<E> {
    pub fn new(eval_pos: E, max_mcts_iterations: usize, exploration_constant: f32) -> Self {
        Self {
            game: InteractivePlay::new(eval_pos, max_mcts_iterations, exploration_constant),
            exit: false,
        }
    }

    /// runs the application's main loop until the user quits
    pub fn run(&mut self, terminal: &mut Tui) -> io::Result<()> {
        while !self.exit {
            terminal.draw(|frame| {
                let snapshot = self.game.snapshot();
                render_app(snapshot, frame.size(), frame.buffer_mut());
            })?;

            if event::poll(Duration::from_millis(100))? {
                self.handle_events()?;
            }
        }
        Ok(())
    }

    /// updates the application's state based on user input
    fn handle_events(&mut self) -> io::Result<()> {
        match event::read()? {
            // it's important to check that the event is a key press event as
            // crossterm also emits key release and repeat events on Windows.
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event)
            }
            _ => {}
        };
        Ok(())
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Char('b') => self.game.make_random_move(0.0),
            KeyCode::Char('m') => self.game.increase_mcts_iters(100),
            KeyCode::Char('n') => self.game.reset_game(),
            KeyCode::Char('r') => self.game.make_random_move(1.0),
            KeyCode::Char('q') => self.exit = true,
            KeyCode::Char('t') => self.game.increase_mcts_iters(1),
            KeyCode::Char('u') => self.game.undo_move(),
            KeyCode::Char('1') => self.game.make_move(0),
            KeyCode::Char('2') => self.game.make_move(1),
            KeyCode::Char('3') => self.game.make_move(2),
            KeyCode::Char('4') => self.game.make_move(3),
            KeyCode::Char('5') => self.game.make_move(4),
            KeyCode::Char('6') => self.game.make_move(5),
            KeyCode::Char('7') => self.game.make_move(6),
            _ => {}
        };
    }
}

fn render_app(snapshot: Snapshot, rect: Rect, buf: &mut Buffer) {
    let inner_rect = render_outer_block(rect, buf);
    let layout = Layout::vertical([
        Constraint::Length(10), // Game
        Constraint::Length(7),  // Snapshot Summary
        Constraint::Fill(1),    // Eval and Policy
        Constraint::Length(11), // Instructions
    ])
    .spacing(1)
    .split(inner_rect);
    let game_rect = layout[0];
    let snapshot_rect = layout[1];
    let eval_and_policy_rect = layout[2];
    let instructions_rect = layout[3];

    render_game(snapshot.pos.clone(), game_rect, buf);
    render_snapshot_summary(&snapshot, snapshot_rect, buf);
    render_eval_and_policy(snapshot.value, &snapshot.policy, eval_and_policy_rect, buf);
    render_instructions(instructions_rect, buf);
}

fn render_outer_block(rect: Rect, buf: &mut Buffer) -> Rect {
    let title = Title::from(" c4a0 - Connect Four Alpha Zero ".bold());
    let outer_block = Block::bordered()
        .title(title.alignment(Alignment::Center))
        .padding(Padding::horizontal(1))
        .border_set(border::THICK);
    let inner_area = outer_block.inner(rect);
    outer_block.render(rect, buf);
    inner_area
}

fn render_game(pos: Pos, rect: Rect, buf: &mut Buffer) {
    let isp0 = pos.ply() % 2 == 0;
    let to_play = match pos.is_terminal_state() {
        Some(TerminalState::PlayerWin) if isp0 => vec![" Blue".blue(), " won".into()],
        Some(TerminalState::PlayerWin) => vec![" Red".red(), " won".into()],
        Some(TerminalState::OpponentWin) if isp0 => vec![" Blue".blue(), " won".into()],
        Some(TerminalState::OpponentWin) => vec![" Red".red(), " won".into()],
        Some(TerminalState::Draw) => vec![" Draw".gray()],
        None if isp0 => vec![" Red".red(), " to play".into()],
        None => vec![" Blue".blue(), " to play".into()],
    };
    Paragraph::new(pos.to_string())
        .block(
            Block::bordered()
                .title(" Board")
                .title_bottom(to_play)
                .padding(Padding::uniform(1)),
        )
        .render(rect, buf);
}

fn render_snapshot_summary(snapshot: &Snapshot, rect: Rect, buf: &mut Buffer) {
    Paragraph::new(vec![
        Line::from(vec![
            "Value: ".into(),
            format!("{:.2}", snapshot.value).yellow().bold(),
        ]),
        Line::from(vec![
            "MCTS iters: ".into(),
            snapshot.n_mcts_iterations.to_string().bold(),
            "/".into(),
            snapshot.max_mcts_iterations.to_string().bold(),
        ]),
        Line::from(if snapshot.bg_thread_running {
            "MCTS running".green()
        } else {
            "MCTS stopped".red()
        }),
    ])
    .block(
        Block::bordered()
            .title(" Snapshot")
            .padding(Padding::uniform(1)),
    )
    .render(rect, buf);
}

fn render_eval_and_policy(pos_value: PosValue, policy: &Policy, rect: Rect, buf: &mut Buffer) {
    let layout = Layout::horizontal([Constraint::Length(10), Constraint::Fill(1)]).split(rect);
    render_eval(pos_value, layout[0], buf);
    render_policy(policy, layout[1], buf);
}

fn render_eval(pos_value: PosValue, rect: Rect, buf: &mut Buffer) {
    let value_max = 1000u64;
    let value = ((pos_value + 1.0) / 2.0 * (value_max as f32)) as u64;
    let bars = vec![Bar::default()
        .label("Eval".into())
        .value(value)
        .text_value(format!("{:.2}", pos_value).into())];
    BarChart::default()
        .data(BarGroup::default().bars(&bars))
        .bar_width(5)
        .bar_gap(2)
        .max(value_max)
        .bar_style(if pos_value >= 0.0 {
            Style::new().red()
        } else {
            Style::new().blue()
        })
        .value_style(Style::new().green().bold())
        .label_style(Style::new().white())
        .block(
            Block::bordered()
                .title(" Eval")
                .padding(Padding::uniform(1)),
        )
        .render(rect, buf);
}

fn render_policy(policy: &Policy, rect: Rect, buf: &mut Buffer) {
    let policy_max = 1000u64;
    let bars = policy
        .iter()
        .enumerate()
        .map(|(i, p)| {
            Bar::default()
                .label(format!("{}", i + 1).into())
                .value((p * (policy_max as f32)) as u64)
                .text_value(format!("{:.2}", p)[1..].into())
        })
        .collect::<Vec<_>>();
    BarChart::default()
        .data(BarGroup::default().bars(&bars))
        .bar_width(5)
        .bar_gap(2)
        .max(policy_max)
        .bar_style(Style::new().yellow())
        .value_style(Style::new().green().bold())
        .label_style(Style::new().white())
        .block(
            Block::bordered()
                .title(" Policy")
                .padding(Padding::uniform(1)),
        )
        .render(rect, buf);
}

fn render_instructions(rect: Rect, buf: &mut Buffer) {
    let instruction_text = vec![
        Line::from(vec!["<1-7>".blue().bold(), " Play Move".into()]),
        Line::from(vec!["<B>".blue().bold(), " Play the best move".into()]),
        Line::from(vec!["<R>".blue().bold(), " Play a random move".into()]),
        Line::from(vec!["<M>".blue().bold(), " More MCTS iterations".into()]),
        Line::from(vec!["<U>".blue().bold(), " Undo last move".into()]),
        Line::from(vec!["<N>".blue().bold(), " New game".into()]),
        Line::from(vec!["<Q>".blue().bold(), " Quit".into()]),
    ];
    Paragraph::new(instruction_text)
        .block(
            Block::bordered()
                .title(" Instructions")
                .padding(Padding::uniform(1)),
        )
        .render(rect, buf);
}
