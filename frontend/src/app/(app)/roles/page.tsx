'use client';
import { useEffect, useState } from 'react';
import { fetchRoles, addRole, deleteRole, Participant } from '@/lib/api';
import { Search, Plus, Trash2, Users, Building, Globe, X } from 'lucide-react';

const ink = '#1A1A18';
const inkSec = '#6B6A66';
const inkMuted = '#9B9891';
const inkFaint = '#B0AEA8';
const borderColor = '#E8E6E1';
const warmBg = '#F7F6F3';
const indigo = '#4F46E5';
const negative = '#DC2626';

const AVATAR_COLORS = ['#4F46E5', '#16A34A', '#D97706', '#DC2626', '#0891B2', '#7C3AED'];

function initials(name: string) {
    return name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
}

function avatarColor(name: string) {
    let h = 0; for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0;
    return AVATAR_COLORS[h % AVATAR_COLORS.length];
}

const ROLE_PRESETS = [
    'CEO', 'CTO', 'CFO', 'COO', 'Product Manager', 'Developer', 'Designer',
    'QA Engineer', 'DevOps Engineer', 'Marketing Manager', 'Sales Executive',
    'Data Analyst', 'Project Manager', 'HR Manager', 'Legal Counsel',
];

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

    return (
        <div style={{ padding: '36px 40px', maxWidth: '960px', margin: '0 auto' }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '28px' }}>
                <div>
                    <h1 style={{
                        fontSize: '24px', fontWeight: 600, marginBottom: '6px',
                        fontFamily: '"Instrument Serif", "Playfair Display", Georgia, serif',
                        color: ink,
                    }}>Team Members</h1>
                    <p style={{ color: inkMuted, fontSize: '14px' }}>
                        Map speaker IDs to real team member profiles for role-aware summaries.
                    </p>
                </div>
                <button onClick={() => setShowForm(s => !s)} style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    padding: '9px 16px', borderRadius: '6px',
                    background: showForm ? '#FFFFFF' : indigo,
                    color: showForm ? inkSec : '#FFFFFF',
                    border: showForm ? `1px solid ${borderColor}` : 'none',
                    fontSize: '13px', fontWeight: 600, cursor: 'pointer',
                    fontFamily: '"DM Sans", system-ui, sans-serif',
                }}>
                    {showForm ? <><X style={{ width: 14, height: 14 }} /> Cancel</> : <><Plus style={{ width: 14, height: 14 }} /> Add Member</>}
                </button>
            </div>

            {/* Add Form */}
            {showForm && (
                <div style={{
                    background: '#FFFFFF', border: `1px solid ${borderColor}`, borderRadius: '12px',
                    padding: '24px', marginBottom: '24px',
                }}>
                    <div style={{ fontSize: '15px', fontWeight: 600, color: ink, marginBottom: '4px' }}>Add Team Member</div>
                    <div style={{ fontSize: '13px', color: inkMuted, marginBottom: '20px' }}>
                        Register a person so their role is used when summarizing meeting content.
                    </div>
                    <form onSubmit={handleAdd}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '14px' }}>
                            <div>
                                <label style={{ fontSize: '13px', fontWeight: 500, color: inkSec, display: 'block', marginBottom: '6px' }}>Full Name *</label>
                                <input className="input-warm" value={form.name}
                                    onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                                    placeholder="e.g. Alice Johnson" />
                            </div>
                            <div>
                                <label style={{ fontSize: '13px', fontWeight: 500, color: inkSec, display: 'block', marginBottom: '6px' }}>Job Title *</label>
                                <input className="input-warm" list="role-presets" value={form.role}
                                    onChange={e => setForm(f => ({ ...f, role: e.target.value }))}
                                    placeholder="e.g. CTO, Product Manager" />
                                <datalist id="role-presets">
                                    {ROLE_PRESETS.map(r => <option key={r} value={r} />)}
                                </datalist>
                            </div>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px', marginBottom: '14px' }}>
                            <div>
                                <label style={{ fontSize: '13px', fontWeight: 500, color: inkSec, display: 'block', marginBottom: '6px' }}>
                                    Department <span style={{ color: inkFaint }}>(optional)</span>
                                </label>
                                <input className="input-warm" value={form.department}
                                    onChange={e => setForm(f => ({ ...f, department: e.target.value }))}
                                    placeholder="e.g. Engineering, Finance" />
                            </div>
                            <div style={{ display: 'flex', alignItems: 'flex-end', paddingBottom: '4px' }}>
                                <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '13px', color: inkSec }}>
                                    <input type="checkbox" checked={form.is_external}
                                        onChange={e => setForm(f => ({ ...f, is_external: e.target.checked }))}
                                        style={{ width: 16, height: 16, accentColor: indigo }} />
                                    External / guest participant
                                </label>
                            </div>
                        </div>
                        {error && <div style={{ color: negative, fontSize: '13px', marginBottom: '12px' }}>{error}</div>}
                        <button type="submit" disabled={submitting} style={{
                            display: 'flex', alignItems: 'center', gap: '6px',
                            padding: '9px 18px', borderRadius: '6px',
                            background: indigo, color: '#FFFFFF', border: 'none',
                            fontSize: '13px', fontWeight: 600, cursor: 'pointer',
                            fontFamily: '"DM Sans", system-ui, sans-serif',
                        }}>
                            {submitting ? 'Adding...' : 'Add Member'}
                        </button>
                    </form>
                </div>
            )}

            {/* Search */}
            <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', alignItems: 'center' }}>
                <div style={{ position: 'relative', flex: 1 }}>
                    <Search style={{ position: 'absolute', left: 12, top: '50%', transform: 'translateY(-50%)', width: 15, height: 15, color: inkFaint }} />
                    <input className="input-warm" value={search} onChange={e => setSearch(e.target.value)}
                        placeholder="Search by name, title, or department..."
                        style={{ paddingLeft: '36px' }} />
                </div>
                <span style={{
                    fontSize: '12px', fontWeight: 600, padding: '6px 12px', borderRadius: '100px',
                    background: warmBg, border: `1px solid ${borderColor}`, color: inkMuted,
                }}>
                    {filtered.length} member{filtered.length !== 1 ? 's' : ''}
                </span>
            </div>

            {/* List */}
            {loading ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '14px' }}>
                    {[0, 1, 2].map(i => <div key={i} className="skeleton" style={{ height: 140, borderRadius: '12px' }} />)}
                </div>
            ) : filtered.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '60px 32px', color: inkMuted }}>
                    <Users style={{ width: 40, height: 40, color: inkFaint, marginBottom: '12px' }} />
                    <div style={{ fontSize: '16px', fontWeight: 600, color: inkSec, marginBottom: '6px' }}>
                        {search ? 'No results found' : 'No team members yet'}
                    </div>
                    <div style={{ fontSize: '13px', color: inkMuted }}>
                        {!search && 'Click "Add Member" to get started'}
                    </div>
                </div>
            ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '14px' }}>
                    {filtered.map(p => (
                        <div key={p.display_name} style={{
                            background: '#FFFFFF', border: `1px solid ${borderColor}`, borderRadius: '12px',
                            padding: '18px', transition: 'box-shadow 0.15s, border-color 0.15s',
                        }}
                            onMouseEnter={e => {
                                (e.currentTarget as HTMLElement).style.boxShadow = '0 4px 12px rgba(0,0,0,0.06)';
                                (e.currentTarget as HTMLElement).style.borderColor = '#D4D2CC';
                            }}
                            onMouseLeave={e => {
                                (e.currentTarget as HTMLElement).style.boxShadow = 'none';
                                (e.currentTarget as HTMLElement).style.borderColor = borderColor;
                            }}>
                            {/* Avatar + Name */}
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '14px' }}>
                                <div style={{
                                    width: 44, height: 44, borderRadius: '50%',
                                    background: avatarColor(p.display_name),
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    fontWeight: 700, fontSize: '15px', color: '#FFFFFF', flexShrink: 0,
                                }}>
                                    {initials(p.display_name)}
                                </div>
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontWeight: 600, fontSize: '14px', color: ink, marginBottom: '3px' }}>{p.display_name}</div>
                                    <span style={{
                                        fontSize: '11px', fontWeight: 600, padding: '2px 8px', borderRadius: '100px',
                                        background: '#EEF2FF', color: indigo,
                                    }}>{p.role}</span>
                                </div>
                            </div>

                            {/* Details */}
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '14px', fontSize: '12px', color: inkMuted }}>
                                {p.department && (
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                        <Building style={{ width: 12, height: 12 }} />
                                        {p.department}
                                    </div>
                                )}
                                {p.is_external && (
                                    <span style={{
                                        display: 'flex', alignItems: 'center', gap: '4px',
                                        fontSize: '11px', fontWeight: 600, padding: '2px 8px', borderRadius: '100px',
                                        background: '#FFFBEB', color: '#D97706',
                                    }}>
                                        <Globe style={{ width: 11, height: 11 }} />
                                        External
                                    </span>
                                )}
                                <div style={{ marginLeft: 'auto', fontSize: '11px', color: inkFaint }}>
                                    Added {new Date(p.added_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                                </div>
                            </div>

                            <div style={{ height: '1px', background: borderColor, margin: '0 0 12px' }} />

                            <button onClick={() => handleDelete(p.display_name)} style={{
                                width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
                                padding: '7px 14px', borderRadius: '6px',
                                background: '#FEF2F2', border: '1px solid rgba(220,38,38,0.15)',
                                color: negative, fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                                fontFamily: '"DM Sans", system-ui, sans-serif',
                            }}>
                                <Trash2 style={{ width: 13, height: 13 }} />
                                Remove Member
                            </button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
