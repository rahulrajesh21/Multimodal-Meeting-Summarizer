const React = require('react');

// Simulate the logic to see if it accidentally writes []
let localStorageData = {
    vela_ai_sessions: JSON.stringify([{ id: '1', title: 'test' }])
};

const localStorage = {
    getItem: (key) => localStorageData[key],
    setItem: (key, val) => { localStorageData[key] = val; },
    removeItem: (key) => { delete localStorageData[key]; }
};

let renderCount = 0;
let pendingState = {};
let state = {
    sessions: [],
    isLoaded: false
};

// ... this is too complex to mock perfectly for React batched updates.
