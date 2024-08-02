use std::io;
use std::io::{stdout, Stdout};

use ratatui::layout::{Constraint, Layout};
use ratatui::widgets::Padding;
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
    Frame, Terminal,
};

use crate::c4r::TerminalState;
use crate::interactive_play::InteractivePlay;
use crate::types::EvalPosT;

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

impl<E: EvalPosT + Send + Sync> App<E> {
    pub fn new(eval_pos: E, max_mcts_iterations: usize, exploration_constant: f32) -> Self {
        Self {
            game: InteractivePlay::new(eval_pos, max_mcts_iterations, exploration_constant),
            exit: false,
        }
    }

    /// runs the application's main loop until the user quits
    pub fn run(&mut self, terminal: &mut Tui) -> io::Result<()> {
        while !self.exit {
            terminal.draw(|frame| self.render_frame(frame))?;
            self.handle_events()?;
        }
        Ok(())
    }

    fn render_frame(&self, frame: &mut Frame) {
        frame.render_widget(self, frame.size());
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
            KeyCode::Char('q') => self.exit(),
            KeyCode::Char('r') => self.reset_board(),
            KeyCode::Char('1') => self.make_move(0),
            KeyCode::Char('2') => self.make_move(1),
            KeyCode::Char('3') => self.make_move(2),
            KeyCode::Char('4') => self.make_move(3),
            KeyCode::Char('5') => self.make_move(4),
            KeyCode::Char('6') => self.make_move(5),
            KeyCode::Char('7') => self.make_move(6),
            _ => {}
        }
    }

    fn exit(&mut self) {
        self.exit = true;
    }

    fn reset_board(&mut self) {
        todo!()
    }

    fn make_move(&mut self, mov: usize) {
        self.game.make_move(mov);
    }
}

impl<E: EvalPosT + Send + Sync> Widget for &App<E> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let snapshot = self.game.snapshot();

        let title = Title::from(" c4a0 - Connect Four Alpha Zero ".bold());
        let outer_block = Block::bordered()
            .title(title.alignment(Alignment::Center))
            .padding(Padding::horizontal(1))
            .border_set(border::THICK);
        let inner_area = outer_block.inner(area);
        outer_block.render(area, buf);

        let layout = Layout::vertical([
            Constraint::Length(10), // Game
            Constraint::Fill(1),    // Spacer
            Constraint::Length(7),  // Instructions
        ])
        .spacing(1)
        .split(inner_area);

        // Game
        let mut pos = snapshot.root_pos.clone();
        let isp0 = pos.ply() % 2 == 0;
        if !isp0 {
            pos = pos.invert();
        }
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
            .render(layout[0], buf);

        // Instructions
        let instruction_text = vec![
            Line::from(vec!["<1-7>".blue().bold(), " Play Move".into()]),
            Line::from(vec!["<R>".blue().bold(), " Restart".into()]),
            Line::from(vec!["<Q>".blue().bold(), " Quit".into()]),
        ];
        Paragraph::new(instruction_text)
            .block(
                Block::bordered()
                    .title(" Instructions")
                    .padding(Padding::uniform(1)),
            )
            .render(layout[2], buf);
    }
}
