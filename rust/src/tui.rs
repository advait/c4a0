use std::io;
use std::io::{stdout, Stdout};
use std::time::Duration;

use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style};
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

use crate::c4r::{CellValue, Pos, TerminalState};
use crate::interactive_play::{InteractivePlay, Snapshot};
use crate::types::{EvalPosT, QValue};

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
    pub fn new(
        eval_pos: E,
        max_mcts_iterations: usize,
        c_exploration: f32,
        c_ply_penalty: f32,
    ) -> Self {
        Self {
            game: InteractivePlay::new(eval_pos, max_mcts_iterations, c_exploration, c_ply_penalty),
            exit: false,
        }
    }

    /// runs the application's main loop until the user quits
    pub fn run(&mut self, terminal: &mut Tui) -> io::Result<()> {
        while !self.exit {
            terminal.draw(|frame| {
                let snapshot = self.game.snapshot();
                draw_app(&snapshot, frame.size(), frame.buffer_mut());
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

fn draw_app(snapshot: &Snapshot, rect: Rect, buf: &mut Buffer) {
    let title = Title::from(" c4a0 - Connect Four AlphaZero ".bold());
    let outer_block = Block::bordered()
        .title(title.alignment(Alignment::Center))
        .padding(Padding::horizontal(1))
        .border_set(border::THICK);
    let inner = outer_block.inner(rect);
    outer_block.render(rect, buf);

    let layout = Layout::vertical([
        Constraint::Length(24), // Game, Evals
        Constraint::Fill(1),    // Policy
        Constraint::Length(11), // Instructions
    ])
    .spacing(1)
    .split(inner);

    draw_game_and_evals(&snapshot, layout[0], buf);
    draw_policy(&snapshot, layout[1], buf);
    draw_instructions(layout[2], buf);
}

fn draw_game_and_evals(snapshot: &Snapshot, rect: Rect, buf: &mut Buffer) {
    let isp0 = snapshot.pos.ply() % 2 == 0;
    let to_play = match snapshot.pos.is_terminal_state() {
        Some(TerminalState::PlayerWin) if isp0 => vec![" Blue".blue(), " won".into()],
        Some(TerminalState::PlayerWin) => vec![" Red".red(), " won".into()],
        Some(TerminalState::OpponentWin) if isp0 => vec![" Blue".blue(), " won".into()],
        Some(TerminalState::OpponentWin) => vec![" Red".red(), " won".into()],
        Some(TerminalState::Draw) => vec![" Draw".gray()],
        None if isp0 => vec![" Red".red(), " to play".into()],
        None => vec![" Blue".blue(), " to play".into()],
    };

    let block = Block::bordered()
        .title(" Game")
        .title_bottom(to_play)
        .padding(Padding::uniform(1));
    let inner = block.inner(rect);
    block.render(rect, buf);

    let layout = Layout::horizontal([Constraint::Length(40), Constraint::Length(18)])
        .spacing(1)
        .split(inner);

    draw_game_grid(&snapshot.pos, layout[0], buf);
    draw_evals(snapshot.q_penalty, snapshot.q_no_penalty, layout[1], buf);
}

fn draw_game_grid(pos: &Pos, rect: Rect, buf: &mut Buffer) {
    let cell_width = 5;
    let cell_height = 3;
    for row in 0..Pos::N_ROWS {
        for col in 0..Pos::N_COLS {
            let cell_rect = Rect::new(
                rect.left() + (col as u16 * cell_width),
                rect.top() + (row as u16 * cell_height),
                cell_width,
                cell_height,
            )
            .intersection(rect);
            draw_game_cell(
                pos.get(Pos::N_ROWS - row - 1, col),
                row,
                col,
                cell_rect,
                buf,
            );
        }
    }

    // Labels below grid
    for col in 0..Pos::N_COLS {
        let label_rect = Rect::new(
            rect.left() + (col as u16 * cell_width),
            rect.top() + (Pos::N_ROWS as u16 * cell_height) + 1,
            cell_width,
            cell_height,
        );
        Paragraph::new(format!("{}", col + 1))
            .centered()
            .bold()
            .render(label_rect, buf);
    }
}

fn draw_game_cell(value: Option<CellValue>, row: usize, col: usize, rect: Rect, buf: &mut Buffer) {
    let bg_style = match value {
        Some(CellValue::Player) => Style::default().bg(Color::Red),
        Some(CellValue::Opponent) => Style::default().bg(Color::Blue),
        None => Style::default(),
    };
    for y in (rect.top() + 1)..rect.bottom() {
        for x in (rect.left() + 1)..rect.right() {
            buf.get_mut(x, y).set_style(bg_style);
        }
    }

    let border_style = Style::default().fg(Color::White);
    let mut set_border = |x, y, ch| {
        buf.get_mut(x, y).set_char(ch).set_style(border_style);
    };

    // Draw horizontal top borders
    for x in rect.left()..rect.right() {
        set_border(x, rect.top(), '─');
    }

    // Draw vertical left borders
    for y in rect.top()..rect.bottom() {
        set_border(rect.left(), y, '│');
    }

    // Top left corners
    set_border(
        rect.left(),
        rect.top(),
        match (row, col) {
            (0, 0) => '┌',
            (0, _c) => '┬',
            (_r, 0) => '├',
            _ => '┼',
        },
    );

    if row == Pos::N_ROWS - 1 {
        // Draw horizontal bottom borders
        for x in rect.left()..rect.right() {
            set_border(x, rect.bottom(), '─');
        }

        // Bottom left corners
        set_border(
            rect.left(),
            rect.bottom(),
            match col {
                0 => '└',
                _ => '┴',
            },
        )
    }

    if col == Pos::N_COLS - 1 {
        // Draw vertical right borders
        for y in rect.top()..rect.bottom() {
            set_border(rect.right(), y, '│');
        }

        // Top right corners
        set_border(
            rect.right(),
            rect.top(),
            match row {
                0 => '┐',
                _ => '┤',
            },
        );

        // Single bottom right corner
        if row == Pos::N_ROWS - 1 {
            set_border(rect.right(), rect.bottom(), '┘');
        }
    }
}

fn draw_evals(q_penalty: QValue, q_no_penalty: QValue, rect: Rect, buf: &mut Buffer) {
    let value_max = 1000u64;
    let q_penalty_u64 = ((q_penalty + 1.0) / 2.0 * (value_max as f32)) as u64;
    let q_no_penalty_u64 = ((q_no_penalty + 1.0) / 2.0 * (value_max as f32)) as u64;
    let bars = vec![
        Bar::default()
            .label("Eval".into())
            .value(q_penalty_u64)
            .text_value(format!("{:.2}", q_penalty).into())
            .style(if q_penalty >= 0.0 {
                Style::new().red()
            } else {
                Style::new().blue()
            }),
        Bar::default()
            .label("Win %".into())
            .value(q_no_penalty_u64)
            .text_value(format!("{:.0}%", q_no_penalty * 100.).into())
            .style(if q_no_penalty >= 0.0 {
                Style::new().red()
            } else {
                Style::new().blue()
            }),
    ];
    BarChart::default()
        .data(BarGroup::default().bars(&bars))
        .bar_width((rect.width - 4) / 2 - 1)
        .bar_gap(2)
        .max(value_max)
        .value_style(Style::new().green().bold())
        .label_style(Style::new().white())
        .render(rect, buf);
}

fn draw_policy(snapshot: &Snapshot, rect: Rect, buf: &mut Buffer) {
    let mcts_status = Line::from(vec![
        " ".into(),
        if snapshot.bg_thread_running {
            "MCTS running: ".green()
        } else {
            "MCTS stopped: ".red()
        },
        snapshot.n_mcts_iterations.to_string().bold(),
        "/".into(),
        snapshot.max_mcts_iterations.to_string().bold(),
    ]);

    let policy_max = 1000u64;
    let bars = snapshot
        .policy
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
                .title_bottom(mcts_status)
                .padding(Padding::uniform(1)),
        )
        .render(rect, buf);
}

fn draw_instructions(rect: Rect, buf: &mut Buffer) {
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
