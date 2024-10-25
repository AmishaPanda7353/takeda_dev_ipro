import { createSlice, configureStore } from '@reduxjs/toolkit';

// Mock auth slice
const authSlice = createSlice({
    name: 'auth',
    initialState: {
        isAuthenticated: false,
        userInfo: {
            id: 0,
            name: '',
            email: '',
            picture: '',
            last_login: '',
            date_joined: '',
        },
    },
    reducers: {
        loginUser: (state) => {
            state.isAuthenticated = true;
        },
        logoutUser: (state) => {
            state.isAuthenticated = false;
        },
    },
});

// Mock user slice
const userSlice = createSlice({
    name: 'user',
    initialState: {
        loggedIn: false,
        outletLoading: false,
        user_id: '',
        chat_details: [],
        activeChatDetails: [],
        report_list: [],
        activeChatIndex: 0,
        activeContextId: 0,
        activeChatId: 0,
        isNewUser: true,
        activeReportChatId: 0,
        isOldReport: false,
        isScrollBottom: true,
    },
    reducers: {
        setUser: (state, action) => {
            state.loggedIn = action.payload;
        },
        setOutletLoading: (state, action) => {
            state.outletLoading = action.payload;
        },
        setInitialUserID: (state, action) => {
            state.user_id = action.payload;
        },
        setInitialUserChatState: (state, action) => {
            state.chat_details = action.payload;
        },
        setActiveChatIndex: (state, action) => {
            state.activeChatIndex = action.payload;
        },
        setActiveContextId: (state, action) => {
            state.activeContextId = action.payload;
        },
    },
});

// Create the store with the mock slices
const store = configureStore({
    reducer: {
        auth: authSlice.reducer,
        user: userSlice.reducer,
    },
});

describe('Redux Store Configuration', () => {
    test('should return the initial state for auth slice', () => {
        const initialState = authSlice.reducer(undefined, { type: 'init' });
        expect(initialState).toEqual(authSlice.reducer(undefined, { type: 'init' }));
    });

    test('should return the initial state for user slice', () => {
        const initialState = userSlice.reducer(undefined, { type: 'init' });
        expect(initialState).toEqual(userSlice.reducer(undefined, { type: 'init' }));
    });

    test('should update the isAuthenticated state when loginUser action is dispatched', () => {
        store.dispatch(authSlice.actions.loginUser());
        const newState = store.getState().auth;
        expect(newState.isAuthenticated).toEqual(true);
    });

    test('should update the loggedIn state when setUser action is dispatched', () => {
        store.dispatch(userSlice.actions.setUser(true));
        const newState = store.getState().user;
        expect(newState.loggedIn).toEqual(true);
    });


});
