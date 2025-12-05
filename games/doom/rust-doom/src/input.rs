use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{HtmlCanvasElement, KeyboardEvent, MouseEvent, window};

// DOOM key codes
pub const KEY_RIGHTARROW: u8 = 0xae;
pub const KEY_LEFTARROW: u8 = 0xac;
pub const KEY_UPARROW: u8 = 0xad;
pub const KEY_DOWNARROW: u8 = 0xaf;
pub const KEY_ESCAPE: u8 = 27;
pub const KEY_ENTER: u8 = 13;
pub const KEY_TAB: u8 = 9;
pub const KEY_FIRE: u8 = 0x80;   // Ctrl
pub const KEY_USE: u8 = 0x81;    // Space
pub const KEY_STRAFE: u8 = 0x82; // Alt
pub const KEY_SPEED: u8 = 0x83;  // Shift

#[derive(Debug, Clone)]
pub enum InputEvent {
    KeyDown(u8),
    KeyUp(u8),
    MouseMove { dx: i32, dy: i32 },
    MouseButton { button: u8, pressed: bool },
}

pub struct InputManager {
    events: Rc<RefCell<Vec<InputEvent>>>,
    _keyboard_down: Closure<dyn FnMut(KeyboardEvent)>,
    _keyboard_up: Closure<dyn FnMut(KeyboardEvent)>,
    _mouse_move: Closure<dyn FnMut(MouseEvent)>,
    _mouse_down: Closure<dyn FnMut(MouseEvent)>,
    _mouse_up: Closure<dyn FnMut(MouseEvent)>,
    _click: Closure<dyn FnMut(MouseEvent)>,
}

fn translate_key(code: &str) -> Option<u8> {
    match code {
        "ArrowLeft" | "KeyA" => Some(KEY_LEFTARROW),
        "ArrowRight" | "KeyD" => Some(KEY_RIGHTARROW),
        "ArrowUp" | "KeyW" => Some(KEY_UPARROW),
        "ArrowDown" | "KeyS" => Some(KEY_DOWNARROW),
        "ControlLeft" | "ControlRight" => Some(KEY_FIRE),
        "Space" => Some(KEY_USE),
        "ShiftLeft" | "ShiftRight" => Some(KEY_SPEED),
        "AltLeft" | "AltRight" => Some(KEY_STRAFE),
        "Escape" => Some(KEY_ESCAPE),
        "Enter" => Some(KEY_ENTER),
        "Tab" => Some(KEY_TAB),
        "Digit1" => Some(b'1'),
        "Digit2" => Some(b'2'),
        "Digit3" => Some(b'3'),
        "Digit4" => Some(b'4'),
        "Digit5" => Some(b'5'),
        "Digit6" => Some(b'6'),
        "Digit7" => Some(b'7'),
        "Digit8" => Some(b'8'),
        "Digit9" => Some(b'9'),
        _ => None,
    }
}

impl InputManager {
    pub fn new(canvas: HtmlCanvasElement) -> Result<InputManager, JsValue> {
        let events = Rc::new(RefCell::new(Vec::new()));
        let document = window().unwrap().document().unwrap();

        // Keyboard down
        let events_clone = events.clone();
        let keyboard_down = Closure::wrap(Box::new(move |e: KeyboardEvent| {
            if let Some(key) = translate_key(&e.code()) {
                events_clone.borrow_mut().push(InputEvent::KeyDown(key));
                e.prevent_default();
            }
        }) as Box<dyn FnMut(KeyboardEvent)>);
        document.add_event_listener_with_callback("keydown", keyboard_down.as_ref().unchecked_ref())?;

        // Keyboard up
        let events_clone = events.clone();
        let keyboard_up = Closure::wrap(Box::new(move |e: KeyboardEvent| {
            if let Some(key) = translate_key(&e.code()) {
                events_clone.borrow_mut().push(InputEvent::KeyUp(key));
            }
        }) as Box<dyn FnMut(KeyboardEvent)>);
        document.add_event_listener_with_callback("keyup", keyboard_up.as_ref().unchecked_ref())?;

        // Mouse move
        let events_clone = events.clone();
        let mouse_move = Closure::wrap(Box::new(move |e: MouseEvent| {
            let dx = e.movement_x();
            let dy = e.movement_y();
            if dx != 0 || dy != 0 {
                events_clone.borrow_mut().push(InputEvent::MouseMove { dx, dy });
            }
        }) as Box<dyn FnMut(MouseEvent)>);
        canvas.add_event_listener_with_callback("mousemove", mouse_move.as_ref().unchecked_ref())?;

        // Mouse down
        let events_clone = events.clone();
        let mouse_down = Closure::wrap(Box::new(move |e: MouseEvent| {
            events_clone.borrow_mut().push(InputEvent::MouseButton {
                button: e.button() as u8,
                pressed: true,
            });
        }) as Box<dyn FnMut(MouseEvent)>);
        canvas.add_event_listener_with_callback("mousedown", mouse_down.as_ref().unchecked_ref())?;

        // Mouse up
        let events_clone = events.clone();
        let mouse_up = Closure::wrap(Box::new(move |e: MouseEvent| {
            events_clone.borrow_mut().push(InputEvent::MouseButton {
                button: e.button() as u8,
                pressed: false,
            });
        }) as Box<dyn FnMut(MouseEvent)>);
        canvas.add_event_listener_with_callback("mouseup", mouse_up.as_ref().unchecked_ref())?;

        // Click for pointer lock
        let canvas_clone = canvas.clone();
        let click = Closure::wrap(Box::new(move |_: MouseEvent| {
            canvas_clone.request_pointer_lock();
        }) as Box<dyn FnMut(MouseEvent)>);
        canvas.add_event_listener_with_callback("click", click.as_ref().unchecked_ref())?;

        Ok(InputManager {
            events,
            _keyboard_down: keyboard_down,
            _keyboard_up: keyboard_up,
            _mouse_move: mouse_move,
            _mouse_down: mouse_down,
            _mouse_up: mouse_up,
            _click: click,
        })
    }

    pub fn poll_events(&self) -> Vec<InputEvent> {
        self.events.borrow_mut().drain(..).collect()
    }
}
