'use client';
import { useEffect, useState } from 'react';
import { fetchRoles, addRole, deleteRole, Participant } from '@/lib/api';

export default function RolesPage() {
    const [participants, setParticipants] = useState<Participant[]>([]);
    const [loading, setLoading] = useState(true);
    const [showForm, setShowForm] = useState(false);
    const [form, setForm] = useState({ name: '', role: '', department: '', is_external: false });
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState('');
    const [search, setSearch] = useState('');

    const load = async () => {
        try { const { participants: p } = await fetchRoles(); setParticipants(p); }
        catch { } finally { setLoading(false); }
    };
    useEffect(() => { load(); }, []);

    const handleAdd = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!form.name || !form.role) { setError('Name and role are required.'); return; }
        setSubmitting(true); setError('');
        try {
            await addRole(form);
            setForm({ name: '', role: '', department: '', is_external: false });
            setShowForm(false);
            load();
        } catch (e: any) { setError(e.message); }
        finally { setSubmitting(false); }
    };

    const handleDelete = async (name: string) => {
        if (!confirm(`Remove ${name}?`)) return;
        try { await deleteRole(name); load(); } catch { }
    };

    const filtered = participants.filter(p =>
        p.display_name.toLowerCase().includes(search.toLowerCase()) ||
        p.role.toLowerCase().includes(search.toLowerCase()) ||
        (p.department || '').toLowerCase().includes(search.toLowerCase())
    );

    const ROLE_PRESETS = [
        'CEO', 'CTO', 'CFO', 'COO', 'Product Manager', 'Developer', 'Designer',
        'QA Engineer', 'DevOps Engineer', 'Marketing Manager', 'Sales Executive',
        'Data Analyst', 'Project Manager', 'HR Manager', 'Legal Counsel',
    ];

    function initials(name: string) {
        return name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
    }

    // Assign a consistent color per person based on name hash
    const AVATAR_COLORS = [
        'linear-gradient(135deg,#7c6ff7,#06b6d4)',
        'linear-gradient(135deg,#f59e0b,#ef4444)',
        'linear-gradient(135deg,#22c55e,#06b6d4)',
        'linear-gradient(135deg,#ec4899,#8b5cf6)',
        'linear-gradient(135deg,#f97316,#facc15)',
        'linear-gradient(135deg,#0ea5e9,#6366f1)',
    ];
    function avatarColor(name: string) {
        let h = 0; for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0;
        return AVATAR_COLORS[h % AVATAR_COLORS.length];
    }

    return (
        <>
            <div className="topbar">
                <span className="topbar-title">👥 Team Members</span>
                <button className="btn btn-primary btn-sm" onClick={() => setShowForm(s => !s)}>
                    {showForm ? '✕ Cancel' : '+ Add Member'}
                </button>
            </div>

            <div className="page">
                {/* Add form */}
                {showForm && (
                    <div className="card" style={{ marginBottom: '28px', borderColor: 'var(--accent-dim)' }}>
                        <div style={{ fontWeight: 600, fontSize: '16px', marginBottom: '4px' }}>Add Team Member</div>
                        <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '20px' }}>
                            Register a person so their role is used when summarizing meeting content.
                        </div>
                        <form onSubmit={handleAdd}>
                            <div className="form-row" style={{ marginBottom: '16px' }}>
                                <div className="form-group">
                                    <label className="form-label">Full Name *</label>
                                    <input className="input" value={form.name}
                                        onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                                        placeholder="e.g. Alice Johnson" />
                                </div>
                                <div className="form-group">
                                    <label className="form-label">Job Title *</label>
                                    <input className="input" list="role-presets" value={form.role}
                                        onChange={e => setForm(f => ({ ...f, role: e.target.value }))}
                                        placeholder="e.g. CTO, Product Manager…" />
                                    <datalist id="role-presets">
                                        {ROLE_PRESETS.map(r => <option key={r} value={r} />)}
                                    </datalist>
                                </div>
                            </div>
                            <div className="form-row" style={{ marginBottom: '16px' }}>
                                <div className="form-group">
                                    <label className="form-label">Department <span style={{ color: 'var(--text-muted)' }}>(optional)</span></label>
                                    <input className="input" value={form.department}
                                        onChange={e => setForm(f => ({ ...f, department: e.target.value }))}
                                        placeholder="e.g. Engineering, Finance…" />
                                </div>
                                <div className="form-group" style={{ justifyContent: 'flex-end' }}>
                                    <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer', marginTop: '22px', fontSize: '14px', color: 'var(--text-secondary)' }}>
                                        <input type="checkbox" checked={form.is_external}
                                            onChange={e => setForm(f => ({ ...f, is_external: e.target.checked }))}
                                            style={{ width: 16, height: 16, accentColor: 'var(--accent)' }} />
                                        External / guest participant
                                    </label>
                                </div>
                            </div>
                            {error && <div style={{ color: 'var(--danger)', fontSize: '13px', marginBottom: '12px' }}>{error}</div>}
                            <button type="submit" className="btn btn-primary" disabled={submitting}>
                                {submitting ? <><span className="spinner" /> Adding…</> : '✅ Add Member'}
                            </button>
                        </form>
                    </div>
                )}

                {/* Search + count */}
                <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', alignItems: 'center' }}>
                    <div style={{ position: 'relative', flex: 1 }}>
                        <span style={{ position: 'absolute', left: '14px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', fontSize: '15px', pointerEvents: 'none' }}>🔍</span>
                        <input className="input" value={search} onChange={e => setSearch(e.target.value)}
                            placeholder="Search by name, title, or department…"
                            style={{ paddingLeft: '38px' }} />
                    </div>
                    <span className="badge badge-muted" style={{ whiteSpace: 'nowrap', fontSize: '13px', padding: '8px 14px' }}>
                        {filtered.length} member{filtered.length !== 1 ? 's' : ''}
                    </span>
                </div>

                {loading ? (
                    <div className="empty-state"><span className="spinner" /></div>
                ) : filtered.length === 0 ? (
                    <div className="empty-state">
                        <div className="empty-icon">👥</div>
                        <div className="empty-title">{search ? 'No results found' : 'No team members yet'}</div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '4px' }}>
                            {!search && 'Click "Add Member" to get started'}
                        </div>
                    </div>
                ) : (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '16px' }}>
                        {filtered.map(p => (
                            <div key={p.display_name} className="card card-sm" style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
                                {/* Top: avatar + name + role */}
                                <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '14px' }}>
                                    <div style={{
                                        width: 50, height: 50, borderRadius: '50%',
                                        background: avatarColor(p.display_name),
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        fontWeight: 700, fontSize: '18px', flexShrink: 0,
                                        boxShadow: '0 2px 12px rgba(0,0,0,0.3)',
                                    }}>
                                        {initials(p.display_name)}
                                    </div>
                                    <div style={{ flex: 1, minWidth: 0 }}>
                                        <div style={{ fontWeight: 600, fontSize: '15px', marginBottom: '4px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{p.display_name}</div>
                                        <span className="badge badge-purple" style={{ fontSize: '12px' }}>{p.role}</span>
                                    </div>
                                </div>

                                {/* Details row */}
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '14px' }}>
                                    {p.department && (
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', color: 'var(--text-secondary)' }}>
                                            🏢 {p.department}
                                        </div>
                                    )}
                                    {p.is_external && <span className="badge badge-yellow">🌐 External</span>}
                                    <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                                        Added {new Date(p.added_at).toLocaleDateString('en-IN', { month: 'short', day: 'numeric', year: 'numeric' })}
                                    </div>
                                </div>

                                <div className="divider" style={{ margin: '0 0 12px' }} />

                                {/* Remove */}
                                <button className="btn btn-danger btn-sm" style={{ width: '100%', justifyContent: 'center' }}
                                    onClick={() => handleDelete(p.display_name)}>
                                    🗑 Remove Member
                                </button>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </>
    );
}
